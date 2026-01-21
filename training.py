# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Helper Functions
# =============================================================================

def compute_local_subspace(client, k_components=5):

    device = client.device
    all_se = []

    for data in client.train_loader.dataset:
        if hasattr(data, 'se_stru_0'):
            all_se.append(data.se_stru_0)


    if not all_se:
        return torch.eye(7).to(device)


    X_stru = torch.cat(all_se, dim=0)


    mean = torch.mean(X_stru, dim=0)
    X_centered = X_stru - mean


    if X_centered.shape[0] < k_components:
        return torch.eye(X_centered.shape[1]).to(device)

    try:

        _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        k = min(k_components, Vh.shape[0])
        U_i = Vh[:k, :].T


        return projection_matrix.detach()
    except Exception as e:
        logger.warning(f"SVD failed for client {client.id}: {e}")
        return torch.eye(X_centered.shape[1]).to(device)


def project_prototype(prototype, target_dim, device):

    p_proj = torch.zeros(target_dim).to(device)
    # Copy prototype into the first 'dim' slots
    p_len = min(prototype.shape[0], target_dim)
    p_proj[:p_len] = prototype[:p_len]
    return p_proj


# =============================================================================
# 2. Phase 1: Structure Optimization Training Loop
# =============================================================================

def train_structure_phase(client, global_prototype, lambda_1, lambda_2, lambda_3):
    model = client.gae
    optimizer = client.gae_optimizer
    model.train()


    if not hasattr(client, 'subspace_proj'):
        client.subspace_proj = compute_local_subspace(client)


    p_proj = project_prototype(global_prototype, client.args.hidden_dim, client.device)

    for data in client.train_loader:
        data = data.to(client.device)
        optimizer.zero_grad()


        recon, z, h_L, h_prime_L = model(data)


        l_rec = F.mse_loss(recon, data.se_stru_0)


        batch_p = p_proj.unsqueeze(0).expand(z.size(0), -1)
        l_align = F.mse_loss(z, batch_p)


        recon_proj = torch.matmul(recon, client.subspace_proj)
        l_consist = F.mse_loss(recon, recon_proj)


        loss = (lambda_1 * l_rec) + (lambda_2 * l_align) + (lambda_3 * l_consist)

        loss.backward()
        optimizer.step()

    return model.encoder.state_dict()


# =============================================================================
# 3. Phase 2: Collaborative Training Loop (Refactored)
# =============================================================================

def train_federated_phase(client):

    client.classifier.train() # This sets training mode for feat_model, stru_model, and heads

    optimizer = client.phase2_optimizer
    device = client.device


    old_stru_params = {
        k: v.clone().detach()
        for k, v in client.stru_model.state_dict().items()
    }

    for data in client.train_loader:
        data = data.to(device)
        optimizer.zero_grad()


        log_probs = client.classifier(data)


        loss = F.nll_loss(log_probs, data.y)

        loss.backward()
        optimizer.step()


    grad_norm_sq = 0.0
    new_stru_params = client.stru_model.state_dict()

    for k in new_stru_params:
        if new_stru_params[k].dtype in [torch.float32, torch.float64]:
            diff = new_stru_params[k] - old_stru_params[k]
            grad_norm_sq += torch.sum(diff ** 2).item()

    grad_norm = np.sqrt(grad_norm_sq)

    return client.stru_model.state_dict(), grad_norm


# =============================================================================
# 4. Evaluation Routine (Refactored)
# =============================================================================

def evaluate_model(client):

    model = client.classifier
    model.eval()

    correct = 0
    total = 0
    device = client.device

    loader = client.test_loader
    if loader is None:
        return 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            
            log_probs = model(data)

            preds = log_probs.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    if total == 0:
        return 0.0

    return correct / total