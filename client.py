# -*- coding: utf-8 -*-

import torch
import copy
import logging
import random
from torch_geometric.loader import DataLoader

# Modified imports to include Classifier
from models import FedCHG_GAE, FeatureModel, StructuralModel, FedCHG_Classifier
from features import get_se_stru
from training import train_structure_phase, train_federated_phase, evaluate_model

# Configure Client Logger
logger = logging.getLogger(__name__)


class FedClient:
    def __init__(self, client_id, dataset, args):
        self.id = client_id
        self.args = args
        self.device = args.device

        # Data Splitting to prevent Data Leakage
        # Shuffle and split into Train (80%) and Test (20%)
        dataset = dataset.shuffle()
        num_train = int(len(dataset) * 0.8)

        # Handle edge case for very small datasets
        if num_train == 0 and len(dataset) > 0:
            num_train = len(dataset)

        self.train_dataset = dataset[:num_train]
        self.test_dataset = dataset[num_train:]

        self.num_samples = len(self.train_dataset)

        # Create separate loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        # Test loader (batch_size can be larger for eval)
        if len(self.test_dataset) > 0:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=args.batch_size,
                shuffle=False
            )
        else:
            self.test_loader = None
            logger.warning(f"Client {self.id}: No testing data available (Dataset too small).")

        # 2. Structural Feature Extraction
        self._extract_features(dataset)

        # 3. Model Initialization placeholders
        self.gae = None

        # Models for Phase 2
        self.feat_model = None
        self.stru_model = None
        self.classifier = None  # Renamed/Changed from classifier_head to classifier wrapper

        # 4. Optimizers
        self.opt_phase1 = None
        self.opt_phase2 = None

    def _extract_features(self, dataset):

        # Process the entire passed dataset (which links to train/test subsets)
        for data in dataset:
            data.se_stru_0 = get_se_stru(data, rw_steps=self.args.rw_steps).to(self.device)

    def get_avg_se_stru(self):

        all_se = [d.se_stru_0 for d in self.train_dataset]
        if not all_se:
            return torch.zeros(7)  # Fallback
        concat_se = torch.cat(all_se, dim=0)
        return torch.mean(concat_se, dim=0).cpu()

    # =========================================================================
    # Phase 1: Structure Optimization (GAE)
    # =========================================================================

    def setup_phase1(self):


        input_dim = self.train_dataset[0].se_stru_0.shape[1]

        self.gae = FedCHG_GAE(
            in_dim=input_dim,
            hid_dim=self.args.hidden_dim,
            dropout=self.args.dropout
        ).to(self.device)

        self.opt_phase1 = torch.optim.Adam(
            self.gae.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.gae_optimizer = self.opt_phase1

    def update_phase1(self, global_encoder_state, global_prototype):
        if self.gae is None:
            self.setup_phase1()


        self.gae.encoder.load_state_dict(global_encoder_state)


        updated_encoder_params = train_structure_phase(
            client=self,
            global_prototype=global_prototype,
            lambda_1=self.args.lambda_1,
            lambda_2=self.args.lambda_2,
            lambda_3=self.args.lambda_3
        )
        return updated_encoder_params

    def finalize_structure(self):
        
        if self.gae is None:
            return

        self.gae.eval()
        with torch.no_grad():

            for data in self.train_dataset:
                data = data.to(self.device)
                z, _ = self.gae.encode(data.se_stru_0, data.edge_index)
                se_prime, _ = self.gae.decode(z, data.edge_index)
                data.se_stru_prime = se_prime.detach().cpu()


            if self.test_dataset:
                for data in self.test_dataset:
                    data = data.to(self.device)
                    z, _ = self.gae.encode(data.se_stru_0, data.edge_index)
                    se_prime, _ = self.gae.decode(z, data.edge_index)
                    data.se_stru_prime = se_prime.detach().cpu()

    # =========================================================================
    # Phase 2: Collaborative Training (Decoupled)
    # =========================================================================

    def setup_phase2(self):

        feat_dim = self.train_dataset.num_features


        self.feat_model = FeatureModel(
            input_dim=feat_dim,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout
        ).to(self.device)

        stru_dim = self.train_dataset[0].se_stru_0.shape[1]
        self.stru_model = StructuralModel(
            input_dim=stru_dim,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout
        ).to(self.device)


        self.classifier = FedCHG_Classifier(
            feat_model=self.feat_model,
            stru_model=self.stru_model,
            hidden_dim=self.args.hidden_dim,
            num_classes=self.train_dataset.num_classes
        ).to(self.device)


        self.opt_phase2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.phase2_optimizer = self.opt_phase2

    def update_phase2(self, global_stru_model_state):
        if self.classifier is None:
            self.setup_phase2()


        self.stru_model.load_state_dict(global_stru_model_state)


        updated_stru_params, grad_norm = train_federated_phase(self)

        return updated_stru_params, grad_norm

    def evaluate(self, global_stru_model_state=None):

        if self.classifier is None or self.test_loader is None:
            return 0.0


        local_state = None
        if global_stru_model_state is not None:
            local_state = copy.deepcopy(self.stru_model.state_dict())
            self.stru_model.load_state_dict(global_stru_model_state)


        acc = evaluate_model(self)


        if local_state is not None:
            self.stru_model.load_state_dict(local_state)

        return acc