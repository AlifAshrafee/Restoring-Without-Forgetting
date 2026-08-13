"""
3x7 qualitative figure: catastrophic forgetting under sequential FT vs
ConDA's adapter-isolation preservation across all six PSF variants V1-V6.

Layout (single test image rendered through six different shift-variant blurs):
                    V1  |  V2 |  V3 |  V4 |  V5 |  V6 |  GT
    Input         : [..]| [..]| [..]| [..]| [..]| [..]| [GT]
    Sequential FT : [..]| [..]| [..]| [..]| [..]| [..]| [GT]  (PSNR overlay)
    ConDA (Ours)  : [..]| [..]| [..]| [..]| [..]| [..]| [GT]  (PSNR + router overlay)

ConDA inference uses the prototype router for adapter selection, so each cell
also reports the predicted variant (->Vk) and a check / cross indicating
whether the router selected the correct adapter for the variant being processed.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conda_yaml",
                   default="./Options/Shift_Variant_V1_V2_V3_V4_V5_ft_V6_Deblurring_Restormer_Adapters.yml")
    p.add_argument("--conda_weights",
                   default="../experiments/archived_checkpoints/V1_V2_V3_V4_V5_ft_V6_Adapters/net_g_latest.pth")
    p.add_argument("--prototypes", 
                   default="../experiments/archived_checkpoints/prototypes/blur_prototypes.pth",
                   help="prototype bank .pth file")
    p.add_argument("--seq_weights",
                   default="../experiments/archived_checkpoints/V1_V2_V3_V4_V5_ft_V6/net_g_latest.pth")
    p.add_argument("--data_root", default="./Datasets/val")
    p.add_argument("--filename", default="0834.png")  # options: 806, 834, 891
    p.add_argument("--crop_size", type=int, default=320)
    p.add_argument("--output",
                   default="../../outputs/forgetting_visualizations/qualitative_forgetting_2.pdf")
    return p.parse_args()


def load_and_crop(path, crop_size):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if crop_size <= 0:
        return img
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Crop size {crop_size} is larger than image dimensions ({h}x{w})")
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    return img[top:top + crop_size, left:left + crop_size]


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()


def to_numpy(t):
    return t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)


def pad_to_factor(t, factor=8):
    _, _, h, w = t.shape
    ph, pw = (factor - h % factor) % factor, (factor - w % factor) % factor
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, h, w


def compute_psnr(pred, gt):
    p = (pred * 255).clip(0, 255).astype(np.uint8)
    g = (gt * 255).clip(0, 255).astype(np.uint8)
    return float(psnr_fn(g, p, data_range=255))


def build_vanilla_restormer(weights_path):
    from basicsr.models.archs.restormer_arch import Restormer
    model = Restormer()
    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint.get("params", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def build_conda_restormer(yaml_file, weights):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_domain_incremental import build_model

    class _A:
        pass

    a = _A()
    a.yaml_file = yaml_file
    a.weights = weights
    return build_model(a).cuda().eval()


class BottleneckExtractor:
    def __init__(self, model):
        self.model = model
        self._features = {}
        self._handle = model.latent[-1].register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self._features["b"] = o

    def extract(self, x):
        with torch.no_grad():
            _ = self.model(x, adapter_id=-1)
        e = F.adaptive_avg_pool2d(self._features["b"], 1).flatten(1)
        return F.normalize(e, p=2, dim=1)

    def close(self):
        self._handle.remove()


def route(emb, prototypes):
    sims = {k: float(torch.dot(emb.squeeze(0), p.to(emb.device)))
            for k, p in prototypes.items()}
    return max(sims, key=sims.get)


def key_to_adapter_id(k):
    n = int(str(k).lstrip("Vv"))
    return n - 2


def infer(model, blur_np, adapter_id=None):
    x, h, w = pad_to_factor(to_tensor(blur_np))
    with torch.no_grad():
        y = model(x) if adapter_id is None else model(x, adapter_id=adapter_id)
    return to_numpy(y[..., :h, :w])


def annotate_psnr(ax, psnr):
    ax.text(0.04, 0.06, f"{psnr:.2f}dB",
            transform=ax.transAxes, fontsize=14, color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4),
            ha="left", va="bottom")


def annotate_routing(ax, pred_key, correct):
    color = "lightgreen" if correct else "salmon"
    marker = "\u2713" if correct else "\u2717"
    ax.text(0.02, 0.96, f"\u2192D{int(str(pred_key).lstrip('Vv'))} {marker}",  # consistency with paper
            transform=ax.transAxes, fontsize=16, color=color,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            ha="left", va="top")


def main():
    args = parse_args()
    variants = list(range(1, 7))

    conda = build_conda_restormer(args.conda_yaml, args.conda_weights)
    seq = build_vanilla_restormer(args.seq_weights)
    prototypes = torch.load(args.prototypes, map_location="cpu")["prototypes"]
    extractor = BottleneckExtractor(conda)

    rows = []
    for v in variants:
        blur = load_and_crop(
            os.path.join(args.data_root, f"ShiftVariant_V{v}_Full_Images",
                         "input_crops", args.filename), args.crop_size)
        gt = load_and_crop(
            os.path.join(args.data_root, f"ShiftVariant_V{v}_Full_Images",
                         "target_crops", args.filename), args.crop_size)

        seq_out = infer(seq, blur)

        x_pad, _, _ = pad_to_factor(to_tensor(blur))
        emb = extractor.extract(x_pad)
        pred_key = route(emb, prototypes)
        adapter_id = key_to_adapter_id(pred_key)
        conda_out = infer(conda, blur, adapter_id=adapter_id)

        seq_psnr = compute_psnr(seq_out, gt)
        conda_psnr = compute_psnr(conda_out, gt)
        correct = int(str(pred_key).lstrip("Vv")) == v

        print(f"V{v}: Seq ={seq_psnr:6.2f}dB | "
              f"ConDA(->V{int(str(pred_key).lstrip('Vv'))} {'✓' if correct else 'X'}) ="
              f"{conda_psnr:6.2f}dB")

        rows.append({
            "v": v, "blur": blur, "gt": gt,
            "seq": seq_out, "seq_psnr": seq_psnr,
            "conda": conda_out, "conda_psnr": conda_psnr,
            "pred_key": pred_key, "correct": correct,
        })

    extractor.close()

    print("\nRendering figure...")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    _, axes = plt.subplots(3, 7, figsize=(15, 6.6))
    row_labels = ["Input", "Sequential FT", "ConDA (Ours)"]
    col_titles = [f"D{v}" for v in variants] + ["GT"]  # D instead of V for consistency with paper

    for r in range(3):
        for c in range(7):
            ax = axes[r, c]
            if c < 6:
                if r == 0:
                    ax.imshow(rows[c]["blur"])
                elif r == 1:
                    ax.imshow(rows[c]["seq"])
                    annotate_psnr(ax, rows[c]["seq_psnr"])
                else:
                    ax.imshow(rows[c]["conda"])
                    annotate_psnr(ax, rows[c]["conda_psnr"])
                    annotate_routing(ax, rows[c]["pred_key"], rows[c]["correct"])
            else:
                ax.imshow(rows[0]["gt"])

            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=16)

        axes[r, 0].set_ylabel(row_labels[r], fontsize=16, rotation=90)

    plt.subplots_adjust(wspace=0.04, hspace=0.06)
    plt.savefig(args.output, format="pdf", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
