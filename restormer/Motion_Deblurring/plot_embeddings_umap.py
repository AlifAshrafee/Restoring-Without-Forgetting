"""
Loads pre-extracted embeddings from a .npz file (output of
extract_embeddings.py) and renders a UMAP scatter.

Output filename auto-encodes UMAP parameters (e.g. umap_n15_d0.5_cosine.pdf)
so successive runs don't clobber each other.

Example:
    python plot_embeddings_umap.py \\
        --input_npz extracted_embeddings.npz \\
        --n_neighbors 10 --min_dist 0.1 --spread 2.0 \\
"""

import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_npz",
                   default='../experiments/archived_checkpoints/extracted_embeddings/extracted_embeddings.npz')
    p.add_argument("--output", default=None,
                   help="output filename; auto-named from UMAP params if omitted")

    # umap parameters
    p.add_argument("--n_neighbors", type=int, default=10)
    p.add_argument("--min_dist", type=float, default=0.3)
    p.add_argument("--spread", type=float, default=2.0)

    # visualization
    p.add_argument("--marker_size", type=float, default=24)
    p.add_argument("--alpha", type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    domain_names = [str(n) for n in data["domain_names"]]
    print(f"Loaded {embeddings.shape[0]} embeddings, "
          f"{len(domain_names)} domains, dim={embeddings.shape[1]}")

    import umap
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, metric="cosine", 
                        min_dist=args.min_dist, spread=args.spread)
    coords = reducer.fit_transform(embeddings)

    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    n = len(domain_names)
    n_synth = sum(1 for nm in domain_names if nm.startswith("V"))
    order = list(range(n_synth)) + list(range(n_synth, n))

    _, ax = plt.subplots(figsize=(7.5, 6.0))
    for i in order:
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                   label= f"D{i+1}" if domain_names[i].startswith("V") else domain_names[i], 
                    s=args.marker_size, alpha=args.alpha, 
                    edgecolors='w', linewidths=0.3)

    ax.set_xlabel("UMAP-1", fontsize=16)
    ax.set_ylabel("UMAP-2", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best")
    plt.tight_layout()

    output = args.output or (
        f"../../outputs/embeddings_visualization/umap_n{args.n_neighbors}_d{args.min_dist}"
        f"_s{args.spread}.pdf"
    )
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
