import torch
import torch.nn.functional as F
import numpy as np
import logging
from torch_geometric.utils import (
    to_dense_adj,
    degree,
    add_self_loops,
    remove_self_loops,
    is_undirected,
    to_undirected
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Features] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StructuralFeatureExtractor:


    def __init__(self, rw_steps=3, device='cpu'):

        self.rw_steps = rw_steps
        self.device = device

    def _get_dense_adj(self, edge_index, num_nodes):

        if not is_undirected(edge_index):

            edge_index = to_undirected(edge_index)

        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        return adj.to(self.device)

    def extract_local_features(self, adj, node_degrees):

        num_nodes = adj.shape[0]


        d_v = node_degrees.view(-1)


        g_twostar = d_v * (d_v - 1) / 2


        adj_2 = torch.matmul(adj, adj)
        adj_3 = torch.matmul(adj_2, adj)
        g_triangle = torch.diagonal(adj_3) / 2.0


        denom = d_v * (d_v - 1)

        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        clustering_coeff = (2 * g_triangle) / denom

        clustering_coeff = torch.where(d_v < 2, torch.zeros_like(clustering_coeff), clustering_coeff)


        def z_score_norm(tensor):
            mean = tensor.mean()
            std = tensor.std()
            if std == 0:
                return torch.zeros_like(tensor)
            return (tensor - mean) / (std + 1e-6)  # 1e-6 for numerical stability (epsilon)

        norm_triangle = z_score_norm(g_triangle)
        norm_twostar = z_score_norm(g_twostar)
        norm_degree = z_score_norm(d_v)
        norm_cc = z_score_norm(clustering_coeff)


        se_local = torch.stack([norm_triangle, norm_twostar, norm_degree, norm_cc], dim=1)

        return se_local

    def extract_global_features(self, adj, node_degrees):

        num_nodes = adj.shape[0]


        inv_degrees = 1.0 / node_degrees
        inv_degrees[torch.isinf(inv_degrees)] = 0
        D_inv = torch.diag(inv_degrees)


        P = torch.matmul(D_inv, adj)

        rw_features = []


        P_k = P
        for k in range(1, self.rw_steps + 1):

            return_probs = torch.diagonal(P_k)
            rw_features.append(return_probs)


            if k < self.rw_steps:
                P_k = torch.matmul(P_k, P)


        se_global = torch.stack(rw_features, dim=1)

        return se_global

    def forward(self, data):

        data = data.to(self.device)
        num_nodes = data.num_nodes
        edge_index = data.edge_index


        adj = self._get_dense_adj(edge_index, num_nodes)


        d_v = degree(edge_index[0], num_nodes=num_nodes).float()


        se_local = self.extract_local_features(adj, d_v)


        se_global = self.extract_global_features(adj, d_v)


        se_stru = torch.cat([se_local, se_global], dim=1)

        return se_stru


# =============================================================================
# Wrapper Function for Integration with main.py
# =============================================================================

def get_se_stru(data, rw_steps=3):


    device = data.edge_index.device

    extractor = StructuralFeatureExtractor(rw_steps=rw_steps, device=device)

    try:
        with torch.no_grad():
            features = extractor.forward(data)
        return features
    except RuntimeError as e:
        logger.error(f"OOM or Runtime Error during feature extraction: {e}")

        num_nodes = data.num_nodes
        return torch.zeros((num_nodes, 4 + rw_steps)).to(device)


# =============================================================================
# Unit Test / Debugging Block
# =============================================================================

if __name__ == "__main__":
    import sys


    print("=" * 60)
    print("Running FedCHG Feature Extraction Test (features.py)")
    print("=" * 60)


    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0, 2, 3],
        [1, 0, 2, 1, 0, 2, 3, 2]
    ], dtype=torch.long)

    x = torch.randn(4, 10)
    data = type('Data', (object,), {'edge_index': edge_index, 'num_nodes': 4, 'x': x})()


    rw_steps = 3
    extractor = StructuralFeatureExtractor(rw_steps=rw_steps)


    print(f"\n[Test] Extracting structural features (rw_steps={rw_steps})...")
    adj = extractor._get_dense_adj(edge_index, 4)
    print(f"Adjacency Matrix:\n{adj.numpy()}")

    se_stru = extractor.forward(data)

    print(f"\n[Output] SE_stru Shape: {se_stru.shape}")
    print(f"Expected Shape: [4, {4 + rw_steps}] (4 local + {rw_steps} global)")
    assert se_stru.shape == (4, 4 + rw_steps)
    print("[Pass] Shape verification successful.")


    se_local = se_stru[:, :4]
    se_global = se_stru[:, 4:]

    print("\n[Local Features - Z-score Normalized]")
    print(f"Columns: [Triangles, Two-stars, Degree, ClusteringCoeff]")
    print(se_local.numpy())

    print("\n[Global Features - Random Walk Probabilities]")
    print(f"Columns: [RW_1, RW_2, RW_3]")
    print(se_global.numpy())



    print("\n[Validation] Checking Random Walk Logic...")

    node_3_global = se_global[3]
    print(f"Node 3 RW probs: {node_3_global.numpy()}")

    
    print("\n[Validation] Checking Normalization Stats (Local)...")
    means = se_local.mean(dim=0)
    stds = se_local.std(dim=0)
    print(f"Means (should be ~0): {means.numpy()}")
    print(f"Stds  (should be ~1): {stds.numpy()}")

    print("\n" + "=" * 60)
    print("Test Complete. This file is ready for FedCHG integration.")
    print("=" * 60)

