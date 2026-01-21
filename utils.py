import os
import sys
import random
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from datetime import datetime


# =============================================================================
# 1. Logging Infrastructure
# =============================================================================

def setup_logger(name, save_dir='./logs', level=logging.INFO):
 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'{name}_{timestamp}.log')


    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)


    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Saving logs to: {log_file}")
    print_system_info(logger)

    return logger


def print_system_info(logger):

    logger.info("=" * 40)
    logger.info("System Configuration:")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes")
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("CUDA Available: No (Using CPU)")
    logger.info("=" * 40)


# =============================================================================
# 2. Mathematical & Metric Utilities (Required by main.py)
# =============================================================================

def flatten_params(state_dict):


    params = []
    for k in sorted(state_dict.keys()):  # Sort keys to ensure consistent order
        v = state_dict[k]
        params.append(v.view(-1))
    return torch.cat(params)


def compute_structural_similarity(local_params, global_params):

    with torch.no_grad():
        vec_local = flatten_params(local_params)
        vec_global = flatten_params(global_params)


        if vec_local.device != vec_global.device:
            vec_global = vec_global.to(vec_local.device)


        similarity = F.cosine_similarity(vec_local.unsqueeze(0), vec_global.unsqueeze(0), eps=1e-8)

    return similarity.item()


def compute_gradient_influence(local_params, global_params):

    with torch.no_grad():
        norm_sq = 0.0
        for k in local_params.keys():
            # Only compute for floating point parameters (weights/biases)
            if local_params[k].dtype in [torch.float32, torch.float64]:
                diff = local_params[k] - global_params[k].to(local_params[k].device)
                norm_sq += torch.sum(diff ** 2).item()

    return np.sqrt(norm_sq)


# =============================================================================
# 3. Statistical Analysis Tools (Paper Section 5.3 & 5.6)
# =============================================================================

def calculate_wasserstein_distance(dist_a, dist_b):

    if isinstance(dist_a, torch.Tensor):
        dist_a = dist_a.cpu().numpy().flatten()
    if isinstance(dist_b, torch.Tensor):
        dist_b = dist_b.cpu().numpy().flatten()

    return wasserstein_distance(dist_a, dist_b)


def conduct_t_test(results_method, results_baseline):

    t_stat, p_val = stats.ttest_ind(results_method, results_baseline, equal_var=False)

    sig_level = "ns"
    if p_val < 0.001:
        sig_level = "***"
    elif p_val < 0.01:
        sig_level = "**"
    elif p_val < 0.05:
        sig_level = "*"

    return {
        't_stat': t_stat,
        'p_value': p_val,
        'significance': sig_level
    }


# =============================================================================
# 4. Visualization Utilities (Reproducing Paper Figures)
# =============================================================================

def plot_convergence_curve(metrics_dict, save_path):

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    for method_name, accuracies in metrics_dict.items():
        rounds = range(1, len(accuracies) + 1)
        # Smoothing (Optional): Moving average
        window_size = 5
        if len(accuracies) > window_size * 2:
            smoothed = np.convolve(accuracies, np.ones(window_size) / window_size, mode='valid')
            rounds_smooth = range(window_size, len(accuracies) + 1)
            plt.plot(rounds_smooth, smoothed, label=method_name, linewidth=2)
        else:
            plt.plot(rounds, accuracies, label=method_name, linewidth=2)

    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Convergence Analysis', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Convergence plot saved to {save_path}")


def plot_heterogeneity_heatmap(matrix, labels, save_path, title="Wasserstein Distance"):

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




def save_checkpoint(state, filename='checkpoint.pth.tar'):

    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):

    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (round {checkpoint.get('round', 0)})")
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None


# =============================================================================
# 6. Unit Tests (Self-Verification)
# =============================================================================

if __name__ == "__main__":
    print("Running utils.py Unit Tests...")

    # 1. Test Similarity Calculation
    print("\n[Test 1] Cosine Similarity")
    model_a = {'l1.weight': torch.tensor([1.0, 0.0]), 'l1.bias': torch.tensor([0.0])}
    model_b = {'l1.weight': torch.tensor([0.0, 1.0]), 'l1.bias': torch.tensor([0.0])}  # Orthogonal
    model_c = {'l1.weight': torch.tensor([-1.0, 0.0]), 'l1.bias': torch.tensor([0.0])}  # Opposite

    sim_ab = compute_structural_similarity(model_a, model_b)
    sim_ac = compute_structural_similarity(model_a, model_c)

    print(f"Sim(A, B) [Exp: 0.0]: {sim_ab:.4f}")
    print(f"Sim(A, C) [Exp: -1.0]: {sim_ac:.4f}")

    # 2. Test Gradient Influence
    print("\n[Test 2] Gradient Influence")
    grad_norm = compute_gradient_influence(model_a, model_b)
    # Dist = sqrt((1-0)^2 + (0-1)^2) = sqrt(2) approx 1.414
    print(f"Influence(A, B) [Exp: 1.414]: {grad_norm:.4f}")

    # 3. Test T-Test
    print("\n[Test 3] T-Test")
    # Generate synthetic data: Mean 80 vs Mean 75
    group1 = np.random.normal(0.80, 0.02, 10)
    group2 = np.random.normal(0.75, 0.02, 10)
    res = conduct_t_test(group1, group2)
    print(f"T-Test Result: {res}")

    # 4. Test Wasserstein
    print("\n[Test 4] Wasserstein Distance")
    d1 = np.array([1, 2, 3])
    d2 = np.array([4, 5, 6])  # Shifted by 3
    wd = calculate_wasserstein_distance(d1, d2)
    print(f"WD([1,2,3], [4,5,6]) [Exp: 3.0]: {wd:.4f}")

    print("\nUtils tests completed.")