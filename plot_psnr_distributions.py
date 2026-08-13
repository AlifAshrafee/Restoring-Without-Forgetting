"""
Plot PSNR distributions across degradation domains to verify severity alignment.

Usage:
    python plot_psnr_distributions.py --data_root ./datasets/multiple_degradations
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

DOMAINS = ["D2_deblur", "D3_derain", "D4_dehaze", "D5_lowlight"]
COLORS = {"D2_deblur": "#1f77b4", "D3_derain": "#ffeb0e",
          "D4_dehaze": "#2ca02c", "D5_lowlight": "#d62728"}
LABELS = {"D2_deblur": "Deblur", "D3_derain": "Derain",
          "D4_dehaze": "Dehaze", "D5_lowlight": "Lowlight"}


def compute_psnr(gt, deg):
    mse = np.mean((gt.astype(np.float64) - deg.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./datasets/multiple_degradations")
    parser.add_argument("--output", type=str, 
                        default="outputs/degradations_sanity_check_output/psnr_distributions.png")
    args = parser.parse_args()

    results = {}

    for domain in DOMAINS:
        gt_dir = os.path.join(args.data_root, domain, "val", "gt")
        deg_dir = os.path.join(args.data_root, domain, "val", "degraded")

        psnrs = []
        for fname in sorted(os.listdir(gt_dir)):
            gt = cv2.imread(os.path.join(gt_dir, fname))
            deg = cv2.imread(os.path.join(deg_dir, fname))
            if gt is not None and deg is not None:
                psnrs.append(compute_psnr(gt, deg))

        results[domain] = np.array(psnrs)
        print(f"{LABELS[domain]:<10s}  mean={np.mean(psnrs):.2f}  "
              f"std={np.std(psnrs):.2f}  min={np.min(psnrs):.2f}  "
              f"max={np.max(psnrs):.2f}  n={len(psnrs)}")

    # plot
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # histograms
    bins = np.linspace(8, 28, 30)
    for domain in DOMAINS:
        ax1.hist(results[domain], bins=bins, alpha=0.5,
                 color=COLORS[domain], label=LABELS[domain], edgecolor="white")
    ax1.set_xlabel("PSNR (dB)")
    ax1.set_ylabel("Count")
    ax1.set_title("PSNR Distribution per Domain")
    ax1.legend()

    # box plot
    data = [results[d] for d in DOMAINS]
    bp = ax2.boxplot(data, tick_labels=[LABELS[d] for d in DOMAINS], patch_artist=True)
    for patch, domain in zip(bp["boxes"], DOMAINS):
        patch.set_facecolor(COLORS[domain])
        patch.set_alpha(0.6)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR Spread per Domain")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
