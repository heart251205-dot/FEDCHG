import argparse
import os
import sys
import time
import copy
import logging
import random
import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_loader import load_cross_domain_data
from features import get_se_stru
from models import FedCHG_GAE, FeatureModel, StructuralModel
from training import train_structure_phase, train_federated_phase, evaluate_model
from server import aggregate_encoders, aggregate_structural_weighted
from utils import setup_logger, compute_structural_similarity
from client import FedClient
from config import Config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="FedCHG: Graph Autoencoder Enhanced FGL")

    defaults = Config.FEDCHG_PARAMS

    parser.add_argument('--scenario', type=str, default='CHEM-BIO', help='Cross-domain scenario')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset storage')
    parser.add_argument('--batch_size', type=int, default=32, help='Local batch size')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of independent runs')


    parser.add_argument('--rounds_struct', type=int, default=defaults['rounds_struct'])
    parser.add_argument('--rounds_fed', type=int, default=defaults['rounds_fed'])
    parser.add_argument('--lr', type=float, default=Config.MODEL_PARAMS['lr'])
    parser.add_argument('--weight_decay', type=float, default=Config.MODEL_PARAMS['weight_decay'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden_dim', type=int, default=Config.MODEL_PARAMS['hidden_dim'])
    parser.add_argument('--dropout', type=float, default=Config.MODEL_PARAMS['dropout'])
    parser.add_argument('--rw_steps', type=int, default=defaults['rw_steps'])


    parser.add_argument('--lambda_1', type=float, default=defaults['lambda_1'])
    parser.add_argument('--lambda_2', type=float, default=defaults['lambda_2'])
    parser.add_argument('--lambda_3', type=float, default=defaults['lambda_3'])
    parser.add_argument('--alpha', type=float, default=defaults['alpha'])
    parser.add_argument('--beta', type=float, default=defaults['beta'])
    parser.add_argument('--gamma', type=float, default=defaults['gamma'])

    return parser.parse_args()


def main():
    args = parse_args()

    logger = setup_logger(f"FedCHG_{args.scenario}", save_dir='./logs')
    logger.info(f"Starting FedCHG Experiment on {args.scenario}")
    logger.info(f"Device: {args.device}")


    accuracies = []

    for run in range(args.n_runs):
        seed = run + 2024
        setup_seed(seed)
        logger.info(f"\n=== Run {run + 1}/{args.n_runs} (Seed: {seed}) ===")


        logger.info("Loading Datasets...")
        try:

            client_datasets = load_cross_domain_data(args.scenario, args.data_root)
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return

        clients = []
        logger.info("Initializing Clients...")
        for i, dataset in enumerate(client_datasets):
            client = FedClient(client_id=i, dataset=dataset, args=args)
            clients.append(client)


        logger.info(f"--- Phase 1: Structure Optimization (T1={args.rounds_struct}) ---")


        total_samples = sum([c.num_samples for c in clients])
        prototype_p = torch.zeros(7).to(args.device)  # 7 is structure dim
        for client in clients:
            prototype_p += client.get_avg_se_stru().to(args.device) * (client.num_samples / total_samples)


        clients[0].setup_phase1()
        global_encoder = copy.deepcopy(clients[0].gae.encoder)

        for round_idx in range(args.rounds_struct):
            encoder_updates = []
            for client in clients:
                w_enc = client.update_phase1(global_encoder.state_dict(), prototype_p)
                encoder_updates.append({'params': w_enc, 'samples': client.num_samples})

            global_encoder = aggregate_encoders(global_encoder, encoder_updates)

        logger.info("Phase 1 Complete. Freezing structures...")
        for client in clients:
            client.finalize_structure()


        logger.info(f"--- Phase 2: Collaborative Training (T2={args.rounds_fed}) ---")

        clients[0].setup_phase2()
        global_struct_model = copy.deepcopy(clients[0].stru_model)

        best_acc_run = 0.0

        for round_idx in range(args.rounds_fed):
            client_updates = []


            for client in clients:
                w_stru, grad_norm = client.update_phase2(global_struct_model.state_dict())
                client_updates.append({
                    'params': w_stru, 'grads': grad_norm, 'size': client.num_samples
                })


            similarities = []
            gradients = []
            for update in client_updates:
                sim = compute_structural_similarity(update['params'], global_struct_model.state_dict())
                similarities.append(sim)
                gradients.append(update['grads'])

            global_struct_model = aggregate_structural_weighted(
                global_struct_model, client_updates, similarities, gradients,
                args.alpha, args.beta, args.gamma
            )

            
            if (round_idx + 1) % 5 == 0:
                avg_acc = 0.0
                for client in clients:
                    acc = client.evaluate(global_struct_model.state_dict())
                    avg_acc += acc
                avg_acc /= len(clients)

                if avg_acc > best_acc_run:
                    best_acc_run = avg_acc

                logger.info(f"Round {round_idx + 1}: Avg Acc = {avg_acc:.4f} (Best: {best_acc_run:.4f})")

        accuracies.append(best_acc_run)
        logger.info(f"Run {run + 1} Finished. Best Accuracy: {best_acc_run:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    logger.info(f"Final Results: Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")


if __name__ == "__main__":
    main()