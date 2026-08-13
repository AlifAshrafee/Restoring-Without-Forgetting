"""
Generates a forgetting plot from a lower-triangular PSNR matrix.

Input format (JSON):
    A lower-triangular matrix where entry [i][j] = PSNR on variant j+1
    after sequential training through variant i+1.

Usage:
    python generate_forgetting_plots.py --input restormer_psnr_matrix.json --output forgetting_plot.pdf
"""

import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path


VARIANT_COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # red/coral
    "#228833",  # green
    "#CCBB44",  # yellow
    "#AA3377",  # purple
    "#66CCEE",  # cyan
    "#CC6600",  # orange
]

VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

AVERAGE_COLOR = "#222222"
AVERAGE_STYLE = "--"


def setup_plot_style():
    plt.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "mathtext.fontset": "cm",
            # Font sizes — calibrated for single-column figures
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            # Lines
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            # Axes
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            # Grid
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            # Ticks
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            # Legend
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_psnr_matrix(filepath):
    with open(filepath, "r") as f:
        matrix = json.load(f)

    # ensure lower-triangular structure
    for i, row in enumerate(matrix):
        assert len(row) == i + 1, (f"Row {i} has {len(row)} entries, expected {i+1}")
    return matrix


def plot_forgetting(psnr_matrix, output_path, figsize=None, y_limits=None):
    setup_plot_style()
    n_variants = len(psnr_matrix)

    if figsize is None:
        width = min(3.5 + 0.3 * n_variants, 7.0)  # single column is ~3.5in, double column ~7in
        figsize = (width, width * 0.65)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x_stages = list(range(1, n_variants + 1))

    for var_idx in range(n_variants):
        y_values = []
        x_values = []

        for stage_idx in range(var_idx, n_variants):
            y_values.append(psnr_matrix[stage_idx][var_idx])
            x_values.append(stage_idx + 1)

        color = VARIANT_COLORS[var_idx % len(VARIANT_COLORS)]
        marker = VARIANT_MARKERS[var_idx % len(VARIANT_MARKERS)]
        label = f"V{var_idx + 1}"

        ax.plot(x_values, y_values, color=color, marker=marker, markeredgecolor="white", 
                markeredgewidth=0.8, label=label, zorder=3)

    avg_values = []
    for stage_idx in range(n_variants):
        stage_psnrs = psnr_matrix[stage_idx]
        avg_values.append(np.mean(stage_psnrs))

    ax.plot(x_stages, avg_values, color=AVERAGE_COLOR, linestyle=AVERAGE_STYLE, 
            linewidth=2.0, marker="*", markersize=8, markeredgecolor="white", 
            markeredgewidth=0.8, label="Avg. Incremental", zorder=4)

    ax.set_xlabel("Sequential Training Stage", labelpad=6)
    ax.set_ylabel("PSNR (dB)", labelpad=6)
    ax.set_xticks(x_stages)

    xtick_labels = []
    for i in range(1, n_variants + 1):
        label = f"V{i}" if i == 1 else f"+V{i}"
        xtick_labels.append(label)
    ax.set_xticklabels(xtick_labels)

    if y_limits:
        ax.set_ylim(y_limits)
    else:
        all_vals = [v for row in psnr_matrix for v in row]
        y_min = min(all_vals) - 5.0
        y_max = max(all_vals) + 2.5
        ax.set_ylim(y_min, y_max)

    ax.set_xlim(0.6, n_variants + 0.4)

    ncol = min(n_variants, 4)
    legend = ax.legend(loc="lower left", ncol=ncol, borderaxespad=0.3)
    legend.get_frame().set_linewidth(0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    print(f"PDF Saved: {output_path}")

    png_path = Path(output_path).with_suffix(".png")
    fig.savefig(str(png_path), dpi=300)
    print(f"PNG Saved: {png_path}")

    plt.close(fig)

    return output_path, str(png_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate continual learning forgetting plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to JSON file containing PSNR matrix")
    parser.add_argument("--output", type=str, default="outputs/forgetting_visualizations/forgetting_plot.pdf", 
                        help="Output file path (default: outputs/forgetting_visualizations/forgetting_plot.pdf)")

    args = parser.parse_args()
    psnr_matrix = load_psnr_matrix(args.input)
    plot_forgetting(psnr_matrix, output_path=args.output)


if __name__ == "__main__":
    main()
