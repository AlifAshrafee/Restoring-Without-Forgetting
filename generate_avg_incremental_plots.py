"""
Generates an average incremental PSNR comparison plot from two
lower-triangular PSNR matrices (RwF adapter vs. sequential fine-tuning).

At each stage T_k the plotted value is:
    AIP(T_k) = (1/k) * sum_{j=1}^{k} PSNR(T_k, j)

i.e. the mean PSNR across all tasks seen so far, evaluated after training
on the k-th task.

Input format (JSON, per method):
    A lower-triangular matrix where entry [i][j] = PSNR on task j+1
    after sequential training through task i+1.

Usage:
    # All four curves (matches paper Fig. 1b): Restormer (blue) + NAFNet (red),
    # adapter (solid) vs. sequential FT (dashed):
    python generate_avg_incremental_plots.py \
        --restormer-adapter degradations_restormer_adapter_psnr_matrix.json \
        --restormer-sequential degradations_restormer_psnr_matrix.json \
        --nafnet-adapter degradations_nafnet_adapter_psnr_matrix.json \
        --nafnet-sequential degradations_nafnet_psnr_matrix.json \
        --output outputs/forgetting_visualizations/degradations_rwf_vs_sequential_avg_incremental.pdf \
        --domain-labels "Noise" "Blur" "Rain" "Haze" "Lowlight"

    # Any subset works too (e.g. NAFNet backbone only):
    python generate_avg_incremental_plots.py \
        --nafnet-adapter nafnet_adapter_psnr_matrix.json \
        --nafnet-sequential nafnet_psnr_matrix.json \
        --output nafnet_only.pdf

Color/style legend:
    RwF-Restormer     -> solid  blue
    Restormer (Seq. FT) -> dashed blue
    RwF-NAFNet        -> solid  red
    NAFNet (Seq. FT)    -> dashed red
"""

import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# Color encodes the backbone (blue = Restormer, red = NAFNet);
# linestyle encodes the method (solid = adapter / RwF, dashed = sequential FT).
RESTORMER_COLOR = "#4477AA"  # blue
NAFNET_COLOR = "#EE6677"     # red

METHOD_STYLES = {
    "restormer_adapter": {
        "color": RESTORMER_COLOR,
        "marker": "o",
        "linestyle": "-",
        "alpha": 1.0,
        "label": "RwF-Restormer (Ours)",
    },
    "restormer_sequential": {
        "color": RESTORMER_COLOR,
        "marker": "o",
        "linestyle": "--",
        "alpha": 0.7,
        "label": "Restormer",
    },
    "nafnet_adapter": {
        "color": NAFNET_COLOR,
        "marker": "s",
        "linestyle": "-",
        "alpha": 1.0,
        "label": "RwF-NAFNet (Ours)",
    },
    "nafnet_sequential": {
        "color": NAFNET_COLOR,
        "marker": "s",
        "linestyle": "--",
        "alpha": 0.7,
        "label": "NAFNet",
    },
}

# (adapter_key, sequential_key, fill_color) pairs whose forgetting gap is shaded
BACKBONE_PAIRS = [
    ("restormer_adapter", "restormer_sequential", RESTORMER_COLOR),
    ("nafnet_adapter", "nafnet_sequential", NAFNET_COLOR),
]

# Fixed draw order so solid adapter lines sit above their dashed counterparts
PLOT_ORDER = [
    "restormer_sequential",
    "nafnet_sequential",
    "restormer_adapter",
    "nafnet_adapter",
]

FILL_ALPHA = 0.12


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
            "legend.fontsize": 9,
            # Lines
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
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

    for i, row in enumerate(matrix):
        assert len(row) == i + 1, f"Row {i} has {len(row)} entries, expected {i + 1}"
    return matrix


def compute_avg_incremental(psnr_matrix):
    return [np.mean(row) for row in psnr_matrix]


def plot_avg_incremental(matrices, output_path,
                         domain_labels=None, figsize=None, y_limits=None):
    """matrices: dict mapping method key -> lower-triangular PSNR matrix (or None)."""
    setup_plot_style()

    provided = {k: m for k, m in matrices.items() if m is not None}
    assert provided, "At least one PSNR matrix must be provided."

    n_stages_set = {len(m) for m in provided.values()}
    assert len(n_stages_set) == 1, f"Matrix size mismatch across methods: {n_stages_set}"
    n_stages = n_stages_set.pop()

    avg = {k: compute_avg_incremental(m) for k, m in provided.items()}

    if figsize is None:
        width = min(3.5 + 0.3 * n_stages, 7.0)
        figsize = (width, width * 0.65)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_stages = list(range(1, n_stages + 1))

    # Per-backbone shaded forgetting gap + delta annotations
    for adp_key, seq_key, fill_color in BACKBONE_PAIRS:
        if adp_key not in avg or seq_key not in avg:
            continue
        a_vals, s_vals = avg[adp_key], avg[seq_key]
        ax.fill_between(x_stages, s_vals, a_vals,
                        where=[a >= s for a, s in zip(a_vals, s_vals)],
                        color=fill_color, alpha=FILL_ALPHA, interpolate=True, zorder=1)
        for k, (a, s) in enumerate(zip(a_vals, s_vals)):
            delta = a - s
            if abs(delta) > 0.05:  # skip negligible gaps
                y_mid = a - 1.25
                ax.annotate(f"Δ{delta:.2f}", xy=(x_stages[k], y_mid),
                            color=fill_color, fontsize=7.5, ha="center", va="center",
                            xytext=(0, 0), textcoords="offset points")

    for key in PLOT_ORDER:
        if key not in avg:
            continue
        style = METHOD_STYLES[key]
        ax.plot(x_stages, avg[key], color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markeredgecolor="white", alpha=style["alpha"],
                markeredgewidth=0.8, label=style["label"], zorder=3)

    ax.set_xlabel("Incremental Training Stage", labelpad=6)
    ax.set_ylabel("Avg. Incremental PSNR (dB)", labelpad=6)
    ax.set_xticks(x_stages)

    if domain_labels and len(domain_labels) == n_stages:
        xtick_labels = [domain_labels[0] if i == 0 else f"+{domain_labels[i]}" for i in range(n_stages)]
    else:
        xtick_labels = [f"D{i + 1}" if i == 0 else f"+D{i + 1}" for i in range(n_stages)]
    ax.set_xticklabels(xtick_labels)

    if y_limits:
        ax.set_ylim(y_limits)
    else:
        all_vals = [v for vals in avg.values() for v in vals]
        y_min = min(all_vals) - 1.5
        y_max = max(all_vals) + 1.5
        ax.set_ylim(y_min, y_max)

    ax.set_xlim(0.6, n_stages + 0.4)

    ncol = 2 if len(avg) > 2 else 1
    legend = ax.legend(loc="best", borderaxespad=0.3, ncol=ncol)
    legend.get_frame().set_linewidth(0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"PDF saved: {output_path}")

    png_path = Path(output_path).with_suffix(".png")
    fig.savefig(str(png_path), dpi=300)
    print(f"PNG saved: {png_path}")

    plt.close(fig)
    return output_path, str(png_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare average incremental PSNR: RwF adapter vs. sequential fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--restormer-adapter", type=str, default=None,
                        help="JSON file: lower-triangular PSNR matrix for RwF-Restormer (solid blue)")
    parser.add_argument("--restormer-sequential", type=str, default=None,
                        help="JSON file: lower-triangular PSNR matrix for sequential FT Restormer (dashed blue)")
    parser.add_argument("--nafnet-adapter", type=str, default=None,
                        help="JSON file: lower-triangular PSNR matrix for RwF-NAFNet (solid red)")
    parser.add_argument("--nafnet-sequential", type=str, default=None,
                        help="JSON file: lower-triangular PSNR matrix for sequential FT NAFNet (dashed red)")
    parser.add_argument("--output", type=str, 
                        default="outputs/forgetting_visualizations/avg_incremental_plot.pdf",
                        help="Output PDF path")
    parser.add_argument("--domain-labels", type=str, nargs="+", default=None, 
                        help='x-tick labels: --domain-labels "Blur" "Denoise" "Dehaze" "Derain" "Low-Light"')
    parser.add_argument("--y-min", type=float, default=None,
                        help="Override y-axis lower bound")
    parser.add_argument("--y-max", type=float, default=None, 
                        help="Override y-axis upper bound")

    args = parser.parse_args()

    arg_to_key = {
        "restormer_adapter": args.restormer_adapter,
        "restormer_sequential": args.restormer_sequential,
        "nafnet_adapter": args.nafnet_adapter,
        "nafnet_sequential": args.nafnet_sequential,
    }
    if not any(arg_to_key.values()):
        parser.error("Provide at least one PSNR matrix "
                     "(--restormer-adapter / --restormer-sequential / "
                     "--nafnet-adapter / --nafnet-sequential).")

    matrices = {k: load_psnr_matrix(p) for k, p in arg_to_key.items() if p is not None}

    y_limits = None
    if args.y_min is not None and args.y_max is not None:
        y_limits = (args.y_min, args.y_max)

    plot_avg_incremental(matrices, output_path=args.output, 
                         domain_labels=args.domain_labels, y_limits=y_limits)


if __name__ == "__main__":
    main()
