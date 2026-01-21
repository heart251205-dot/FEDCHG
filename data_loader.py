import os
import sys
import torch
import numpy as np
import random
import logging
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
import networkx as nx
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataset(torch.utils.data.Dataset):


    def __init__(self, data_list):
        self._data_list = data_list

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        # Support slicing which returns a subset Dataset
        if isinstance(idx, slice):
            return SimpleDataset(self._data_list[idx])
        elif isinstance(idx, list) or isinstance(idx, torch.Tensor):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return SimpleDataset([self._data_list[i] for i in idx])
        return self._data_list[idx]

    def shuffle(self):
        random.shuffle(self._data_list)
        return self

    @property
    def num_features(self):
        if len(self._data_list) > 0:
            return self._data_list[0].num_features
        return 0

    @property
    def num_classes(self):
        # Calculate num_classes dynamically based on labels in the list
        if not self._data_list:
            return 0
        ys = [d.y.item() for d in self._data_list if d.y is not None]
        if not ys:
            return 0
        return max(ys) + 1


class CrossDomainDataLoader:
    def __init__(self, data_root='./data', max_degree=500):
        self.data_root = data_root
        self.max_degree = max_degree
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

    def _generate_dgraph_subgraphs(self, save_path, num_subgraphs=1000):
        logger.warning("Generating SYNTHETIC DGraph subgraphs for reproduction testing...")

        data_list = []
        for _ in range(num_subgraphs):
            num_nodes = np.random.randint(20, 100)
            g = nx.barabasi_albert_graph(num_nodes, 3)
            data = from_networkx(g)

            # Feature dim 17 matches DGraph-Fin
            data.x = torch.randn(num_nodes, 17)
            y_val = 1 if np.random.random() < 0.01 else 0
            data.y = torch.tensor([y_val], dtype=torch.long)

            data_list.append(data)

        torch.save(data_list, save_path)
        logger.info(f"Saved {num_subgraphs} synthetic DGraph subgraphs to {save_path}")
        return data_list

    def get_dataset(self, name):
        try:
            pre_transform = None
            social_datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']

            tud_name = name
            if name == 'Reddit':
                tud_name = 'REDDIT-BINARY'
                pre_transform = T.OneHotDegree(self.max_degree)
            elif name in social_datasets:
                pre_transform = T.OneHotDegree(self.max_degree)

            if name == 'DGraph':
                dgraph_dir = os.path.join(self.data_root, 'FIN-SOC', 'DGraph')
                if not os.path.exists(dgraph_dir):
                    os.makedirs(dgraph_dir)

                dgraph_path = os.path.join(dgraph_dir, 'processed_dgraph.pt')

                if os.path.exists(dgraph_path):
                    logger.info(f"Loading pre-processed DGraph from {dgraph_path}")
                    data_list = torch.load(dgraph_path)
                    # Important: Wrap in our custom SimpleDataset
                    return SimpleDataset(data_list)
                else:
                    logger.info(f"Processed DGraph not found at {dgraph_path}. Generating...")
                    data_list = self._generate_dgraph_subgraphs(dgraph_path)
                    return SimpleDataset(data_list)

            # Load TUDataset
            dataset = TUDataset(
                root=os.path.join(self.data_root, name),
                name=tud_name,
                pre_transform=pre_transform,
                use_node_attr=True
            )

            if dataset.num_features == 0:
                dataset.transform = T.Constant(value=1)

            # Convert PyG InMemoryDataset to a list of Data objects for our SimpleDataset
            # This ensures modification persistence
            data_list = [data for data in dataset]

            logger.info(f"Loaded {name}: {len(data_list)} graphs, {data_list[0].num_features} features.")
            return SimpleDataset(data_list)

        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            raise e

    def split_dataset(self, dataset, num_parts):
        num_samples = len(dataset)
        # Use range indices for splitting logic
        indices = np.random.permutation(num_samples)
        part_size = int(np.ceil(num_samples / num_parts))

        split_datasets = []
        for i in range(num_parts):
            start = i * part_size
            end = min((i + 1) * part_size, num_samples)
            if start >= end:
                break

            subset_indices = indices[start:end]
            # SimpleDataset supports list/tensor indexing in __getitem__
            subset = dataset[subset_indices]
            split_datasets.append(subset)

        return split_datasets

    def load_scenario(self, scenario_name):
        if scenario_name not in Config.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_config = Config.SCENARIOS[scenario_name]
        target_datasets = scenario_config['datasets']
        total_clients_needed = scenario_config['num_clients']
        split_mode = scenario_config.get('split_mode', False)

        logger.info(f"Loading Scenario: {scenario_name} | Datasets: {target_datasets} | Split Mode: {split_mode}")

        loaded_datasets = []

        if split_mode:
            num_datasets = len(target_datasets)
            if total_clients_needed % num_datasets != 0:
                logger.warning(
                    f"Total clients {total_clients_needed} not divisible by datasets {num_datasets}. Allocation may be uneven.")

            clients_per_dataset = total_clients_needed // num_datasets

            for name in target_datasets:
                dataset = self.get_dataset(name)
                splits = self.split_dataset(dataset, clients_per_dataset)
                loaded_datasets.extend(splits)

            if len(loaded_datasets) < total_clients_needed:
                logger.warning(
                    f"Warning: Created {len(loaded_datasets)} clients, but config asked for {total_clients_needed}.")
        else:
            for name in target_datasets:
                dataset = self.get_dataset(name)
                loaded_datasets.append(dataset)

        return loaded_datasets


def load_cross_domain_data(scenario_name, root='./data'):
    loader = CrossDomainDataLoader(data_root=root)
    datasets = loader.load_scenario(scenario_name)
    return datasets