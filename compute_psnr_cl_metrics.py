"""
Usage:
    python compute_psnr_cl_metrics.py \
        --psnr_matrix restormer_psnr_matrix.json
"""

import argparse
import json
import numpy as np


def load_matrix(filepath):
    with open(filepath, "r") as f:
        matrix = json.load(f)

    for i, row in enumerate(matrix):
        assert len(row) == i + 1, f"Row {i} has {len(row)} entries, expected {i + 1} (lower-triangular format)"
    return matrix


def compute_metrics(matrix):
    B = len(matrix)

    # A_B (Last): (1/B) * Σ_{j=1}^{B} R[B, j] (mean of the final row)
    # Average performance after the final training stage B.
    # Reflects the quality of the final model across all domains.
    a_last = np.mean(matrix[-1])

    # Ā (Avg): (1/B) * Σ_{b=1}^{B} A_b, where A_b = (1/b) * Σ_{j=1}^{b} R[b, j] (mean of per-stage averages A_b)
    # Average incremental performance across all stages.
    # Captures how well performance is maintained throughout sequential training.
    a_avg = np.mean([np.mean(matrix[b]) for b in range(B)])

    # F (Forgetting): (1/(B-1)) * Σ_{j=1}^{B-1} (R[j, j] - R[B, j]) (mean drop from peak to final)
    # Average performance drop on earlier domains after training on subsequent ones.
    # Higher F = more forgetting. F = 0 means no degradation on prior domains.
    if B >= 2:
        forgetting = np.mean([matrix[j][j] - matrix[-1][j] for j in range(B - 1)])
    else:
        forgetting = 0.0

    return {"a_last": a_last, "a_avg": a_avg, "forgetting": forgetting, "n_variants": B}


def print_metrics(psnr_metrics):
    B = psnr_metrics["n_variants"]

    print(f"Continual Learning Metrics ({B} domains)")
    print("-" * 40)
    print(f"{'Metric':<14} {'PSNR (dB)':>12}")
    print("-" * 40)
    print(f"{'A_B (Last)':<14} {psnr_metrics['a_last']:>12.2f}")
    print(f"{'Ā (Average)':<14} {psnr_metrics['a_avg']:>12.2f}")
    print(f"{'F (Forgetting)':<14} {psnr_metrics['forgetting']:>12.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CL metrics from PSNR matrix")
    parser.add_argument("--psnr_matrix", type=str, required=True,
                        help="Path to JSON file containing PSNR matrix")

    args = parser.parse_args()
    psnr_matrix = load_matrix(args.psnr_matrix)
    psnr_metrics = compute_metrics(psnr_matrix)
    print_metrics(psnr_metrics)
