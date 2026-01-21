
import os
import sys
import argparse
import subprocess
import logging
import time
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from config import Config
except ImportError:
    print("Error: Could not import 'src.config'. Please ensure 'src/config.py' exists.")
    sys.exit(1)


# =============================================================================
# 1. Logger Setup
# =============================================================================

def setup_runner_logger(log_dir: str) -> logging.Logger:

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'runner_{timestamp}.log')

    logger = logging.getLogger("FedCHG_Runner")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers


    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))  # Cleaner console output
    logger.addHandler(ch)

    return logger


# =============================================================================
# 2. System & Hardware Check
# =============================================================================

def check_environment(gpu_id: int, logger: logging.Logger):

    logger.info("=" * 60)
    logger.info("Step 1: Environment & Hardware Check")
    logger.info("-" * 60)


    logger.info(f"[*] Python Version: {sys.version.split()[0]}")


    try:
        import torch
        logger.info(f"[*] PyTorch Version: {torch.__version__}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"[*] CUDA Available: Yes ({device_count} devices)")
            if 0 <= gpu_id < device_count:
                gpu_name = torch.cuda.get_device_name(gpu_id)
                logger.info(f"[*] Selected GPU: ID {gpu_id} ({gpu_name})")
            else:
                logger.warning(f"[!] Warning: GPU ID {gpu_id} not found. Defaulting to CPU.")
        else:
            logger.warning("[!] Warning: CUDA not available. Running on CPU (This will be slow).")
    except ImportError:
        logger.error("[x] Error: PyTorch not installed. Please install dependencies from requirements.txt")
        sys.exit(1)


    required_dirs = ['src', 'data']
    for d in required_dirs:
        if not os.path.exists(d):
            if d == 'data':
                logger.info(f"[*] 'data' directory missing. It will be created by data_loader.")
            else:
                logger.error(f"[x] Critical: Directory '{d}' not found.")
                sys.exit(1)

    logger.info("[*] Environment check passed.\n")


# =============================================================================
# 3. Experiment Execution Engine
# =============================================================================

def construct_command(args, scenario: str, log_file: str) -> List[str]:

    cmd = [
        sys.executable,
        "src/main.py",
        "--scenario", scenario,
        "--n_runs", str(args.n_runs),
        "--device", f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--gamma", str(args.gamma)
    ]
    return cmd


def run_scenario(scenario: str, args, logger: logging.Logger, log_dir: str) -> Tuple[bool, str, str]:
    
    logger.info("=" * 60)
    logger.info(f"Step 2: Executing Scenario - {scenario}")
    try:
        desc = Config.SCENARIOS[scenario].get('description', 'N/A')
    except:
        desc = 'N/A'
    logger.info(f"Description: {desc}")
    logger.info(f"Config: Runs={args.n_runs}, α={args.alpha}, β={args.beta}, γ={args.gamma}")
    logger.info("-" * 60)

    scenario_log_name = f"{scenario}_runs{args.n_runs}_alpha{args.alpha}.log"
    scenario_log_path = os.path.join(log_dir, scenario_log_name)

    cmd = construct_command(args, scenario, scenario_log_path)

    logger.info(f"[*] Command: {' '.join(cmd)}")
    logger.info(f"[*] Logging detailed output to: {scenario_log_path}")
    logger.info("[*] Status: Running... (This may take a while)")

    start_time = time.time()

    with open(scenario_log_path, "w") as f_out:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env=os.environ.copy()
            )
            exit_code = process.wait()
        except KeyboardInterrupt:
            logger.warning("\n[!] User interrupted the process.")
            process.kill()
            return False, "Interrupted", scenario_log_path
        except Exception as e:
            logger.error(f"\n[x] Execution failed: {e}")
            return False, f"Error: {str(e)}", scenario_log_path

    duration = time.time() - start_time

    if exit_code == 0:
        logger.info(f"[*] Finished successfully in {duration / 60:.2f} minutes.")
        return True, "Success", scenario_log_path
    else:
        logger.error(f"[x] Process exited with error code {exit_code}. Check log.")
        return False, f"Failed (Code {exit_code})", scenario_log_path


# =============================================================================
# 4. Result Parsing & Reporting
# =============================================================================

def parse_results_from_log(log_path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(log_path):
        return None, None

    mean_acc, std_acc = None, None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in reversed(lines):
            match = re.search(r"Mean Accuracy:\s*([\d\.]+)\s*%\s*±\s*([\d\.]+)\s*%", line)
            if match:
                mean_acc = float(match.group(1))
                std_acc = float(match.group(2))
                break
    except Exception as e:
        print(f"Error parsing log {log_path}: {e}")

    return mean_acc, std_acc


def print_summary_table(results: List[Dict], logger: logging.Logger):
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EXPERIMENTAL RESULTS (Replicating Table 4)")
    logger.info("=" * 80)
    header = f"{'Scenario':<25} | {'FedCHG Accuracy (Mean ± Std)':<30} | {'Status':<10}"
    logger.info(header)
    logger.info("-" * 80)

    for res in results:
        scenario = res['scenario']
        status = res['status']
        if res['mean'] is not None:
            acc_str = f"{res['mean']:.2f}% ± {res['std']:.2f}%"
        else:
            acc_str = "N/A"
        row = f"{scenario:<25} | {acc_str:<30} | {status:<10}"
        logger.info(row)
    logger.info("=" * 80)


# =============================================================================
# 5. Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FedCHG Reproducibility Runner")
    parser.add_argument('--scenario', type=str, default='all', help='Scenario name or "all".')
    parser.add_argument('--n_runs', type=int, default=10, help='Runs per scenario.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID.')
    parser.add_argument('--log_dir', type=str, default='./logs_experiment', help='Log directory.')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.2)

    args = parser.parse_args()
    logger = setup_runner_logger(args.log_dir)
    check_environment(args.gpu, logger)

    if args.scenario.lower() == 'all':
        target_scenarios = list(Config.SCENARIOS.keys())
    elif args.scenario in Config.SCENARIOS:
        target_scenarios = [args.scenario]
    else:
        logger.error(f"Scenario '{args.scenario}' not found.")
        sys.exit(1)

    results_summary = []
    try:
        for i, scenario in enumerate(target_scenarios):
            logger.info(f"\nProcessing {i + 1}/{len(target_scenarios)}: {scenario}")
            success, status_msg, log_path = run_scenario(scenario, args, logger, args.log_dir)
            mean, std = None, None
            if success:
                mean, std = parse_results_from_log(log_path)
            results_summary.append({'scenario': scenario, 'mean': mean, 'std': std, 'status': "OK" if mean else "Fail"})
            time.sleep(2)
    except Exception as e:
        logger.critical(f"[x] Critical Failure: {e}")
    finally:
        print_summary_table(results_summary, logger)

if __name__ == "__main__":
    main()