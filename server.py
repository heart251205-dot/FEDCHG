import torch
import torch.nn.functional as F
import numpy as np
import copy
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Server] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Global Structure Prototype Calculation (Phase 1)
# =============================================================================

def calculate_global_prototype(clients):

    logger.info("Calculating Global Structure Prototype (P)...")

    total_samples = 0
    weighted_sum_se = None
    device = clients[0].args.device

    for client in clients:

        client_se = client.get_avg_se_stru().to(device)
        n_i = client.num_samples

        if weighted_sum_se is None:
            weighted_sum_se = client_se * n_i
        else:
            weighted_sum_se += client_se * n_i

        total_samples += n_i


    global_prototype = weighted_sum_se / (total_samples + 1e-9)

    logger.info(f"Global Prototype Calculated. Norm: {torch.norm(global_prototype):.4f}")
    return global_prototype


# =============================================================================
# 2. Basic Aggregation (Phase 1 - Encoder)
# =============================================================================

def aggregate_encoders(global_encoder, client_updates):

    total_samples = sum([u['samples'] for u in client_updates])
    new_params = copy.deepcopy(client_updates[0]['params'])


    for key in new_params.keys():
        new_params[key] = torch.zeros_like(new_params[key].float())


    for update in client_updates:
        weight = update['samples'] / total_samples
        local_params = update['params']
        for key in new_params.keys():
            new_params[key] += local_params[key] * weight

    global_encoder.load_state_dict(new_params)
    return global_encoder


# =============================================================================
# 3. Weighted Aggregation Utilities (Phase 2)
# =============================================================================

def flatten_params(state_dict):

    # Concatenate all tensors in the state_dict
    return torch.cat([param.view(-1) for param in state_dict.values()])


def compute_cosine_similarity(local_params_dict, global_params_dict):

    # 1. Flatten parameters
    vec_local = flatten_params(local_params_dict)
    vec_global = flatten_params(global_params_dict)

    # 2. Compute Cosine Similarity
    # dim=0 because vectors are 1D
    similarity = F.cosine_similarity(vec_local.unsqueeze(0), vec_global.unsqueeze(0)).item()

    return similarity


def normalize_weights(weights):

    weights = np.array(weights)
    sum_w = np.sum(weights)
    if sum_w == 0:
        return np.ones_like(weights) / len(weights)
    return weights / sum_w


# =============================================================================
# 4. Advanced Weighted Aggregation (Phase 2 - Structural Model)
# =============================================================================

def aggregate_structural_weighted(global_model, client_updates, similarities, gradients, alpha, beta, gamma):

    num_clients = len(client_updates)
    total_samples = sum([u['size'] for u in client_updates])




    w_sizes = [u['size'] / total_samples for u in client_updates]


    raw_sims = np.array(similarities)

    raw_sims = np.nan_to_num(raw_sims, nan=0.0)


    w_sims = F.softmax(torch.tensor(raw_sims), dim=0).numpy()


    raw_grads = np.array(gradients)
    raw_grads = np.nan_to_num(raw_grads, nan=0.0)


    if np.sum(raw_grads) == 0:
        w_grads = np.ones(num_clients) / num_clients
    else:
        w_grads = raw_grads / np.sum(raw_grads)



    final_weights = []
    log_info = []

    for i in range(num_clients):


        score_i = (alpha * w_sizes[i]) + (beta * w_sims[i]) + (gamma * w_grads[i])
        final_weights.append(score_i)

        log_info.append(f"C{i}: Sz={w_sizes[i]:.2f}, Sim={w_sims[i]:.2f}, Gr={w_grads[i]:.2f} -> W={score_i:.2f}")


    final_weights = normalize_weights(final_weights)


    logger.debug(f"Aggregation Weights: {', '.join(log_info[:5])} ...")


    avg_params = copy.deepcopy(client_updates[0]['params'])
    for key in avg_params.keys():
        avg_params[key] = torch.zeros_like(avg_params[key].float())


    for i, update in enumerate(client_updates):
        local_params = update['params']
        weight = final_weights[i]

        for key in avg_params.keys():

            if local_params[key].dtype != avg_params[key].dtype:
                avg_params[key] += local_params[key].float() * weight
            else:
                avg_params[key] += local_params[key] * weight


    global_model.load_state_dict(avg_params)

    return global_model


# =============================================================================
# Supplementary: Server Logic Class (Optional Wrapper)
# =============================================================================

class FedCHGServer:


    def __init__(self, args, input_dim, hidden_dim):
        self.args = args
        self.device = args.device


        from models import FedCHG_GAE, StructuralModel

        self.global_encoder = FedCHG_GAE(in_dim=input_dim, hid_dim=hidden_dim).encoder.to(self.device)
        self.global_struct_model = StructuralModel(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)

        self.round_history = []

    def log_round_metrics(self, round_idx, weights):

        self.round_history.append({
            'round': round_idx,
            'weights': weights
        })




if __name__ == "__main__":

    print("Testing Weighted Aggregation Logic...")


    p1 = {'w': torch.tensor([1.0, 2.0])}
    p2 = {'w': torch.tensor([2.0, 4.0])}  # p2 is 2x p1

    updates = [
        {'params': p1, 'size': 100, 'grads': 0.5},
        {'params': p2, 'size': 100, 'grads': 1.0}
    ]



    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


    global_model = MockModel()


    sims = [1.0, 1.0]
    grads = [0.5, 1.0]



    new_model = aggregate_structural_weighted(
        global_model, updates, sims, grads,
        alpha=0.33, beta=0.33, gamma=0.33
    )

    print(f"Aggregated Params: {new_model.state_dict()['w']}")
   