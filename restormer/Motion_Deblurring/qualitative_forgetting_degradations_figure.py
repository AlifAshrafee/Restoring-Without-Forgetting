"""
3x6 qualitative figure: catastrophic forgetting under sequential FT vs
RwF's adapter-isolation preservation across all five Degradations D1-D5.

Layout (single test image rendered through five different degradations):
                    D1  |  D2 |  D3 |  D4 |  D5 |  GT
    Input         : [..]| [..]| [..]| [..]| [..]| [GT]
    Sequential FT : [..]| [..]| [..]| [..]| [..]| [GT]  (PSNR overlay)
    RwF (Ours)  : [..]| [..]| [..]| [..]| [..]| [GT]  (PSNR + router overlay)

RwF inference uses the prototype router for adapter selection, so each cell
also reports the predicted degradation (->Dk) and a check / cross indicating
whether the router selected the correct adapter for the degradation being processed.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


DOMAIN_TO_VAL_DIR = {
    "D1_denoise": "D1_denoise_Full_Images",
    "D2_deblur": "D2_deblur_Full_Images",
    "D3_derain": "D3_derain_Full_Images",
    "D4_dehaze": "D4_dehaze_Full_Images",
    "D5_lowlight": "D5_lowlight_Full_Images",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rwf_yaml",
                   default="./Options/Degradations_D1_D2_D3_D4_ft_D5_Restormer_Adapters.yml")
    p.add_argument("--rwf_weights",
                   default="../experiments/archived_checkpoints/" \
                   "Degradations_D1_D2_D3_D4_ft_D5_Adapters/net_g_latest.pth")
    p.add_argument("--prototypes", 
                   default="../experiments/archived_checkpoints/prototypes/proto_enc1_v2.pth")
    p.add_argument("--seq_weights",
                   default="../experiments/archived_checkpoints/" \
                   "Degradations_D1_D2_D3_D4_ft_D5/net_g_latest.pth")
    p.add_argument("--data_root", default="./Datasets/val")
    p.add_argument("--filename", default="0834.png")  # options: 806, 834, 891
    p.add_argument("--crop_size", type=int, default=320)
    p.add_argument("--output",
                   default="../../outputs/forgetting_visualizations/" \
                   "qualitative_degradations_forgetting_3.pdf")
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


def build_rwf_restormer(yaml_file, weights):
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
        self._handle = model.encoder_level1[-1].register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self._features["b"] = o

    def extract(self, x):
        with torch.no_grad():
            _ = self.model(x, adapter_id=-1)

        emb_mean = F.adaptive_avg_pool2d(self._features["b"], 1).flatten(1)
        emb_mean_n = F.normalize(emb_mean, p=2, dim=1)
        emb_std  = self._features["b"].flatten(2).std(dim=2)
        emb_std_n  = F.normalize(emb_std,  p=2, dim=1)
        emb = F.normalize(torch.cat([emb_mean_n, emb_std_n], dim=1), p=2, dim=1)
        return emb.squeeze(0)

    def close(self):
        self._handle.remove()


def route(emb, prototypes):
    sims = {k: float(torch.dot(emb.squeeze(0), p.to(emb.device)))
            for k, p in prototypes.items()}
    return max(sims, key=sims.get)


def infer(model, blur_np, adapter_id=None):
    x, h, w = pad_to_factor(to_tensor(blur_np))
    with torch.no_grad():
        y = model(x) if adapter_id is None else model(x, adapter_id=adapter_id)
    return to_numpy(y[..., :h, :w])


def annotate_psnr(ax, psnr):
    ax.text(0.02, 0.04, f"{psnr:.2f} dB",
            transform=ax.transAxes, fontsize=9, color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            ha="left", va="bottom")


def annotate_routing(ax, pred_key, correct):
    color = "lightgreen" if correct else "salmon"
    marker = "\u2713" if correct else "\u2717"
    ax.text(0.02, 0.96, f"\u2192D{int(str(pred_key).lstrip('Dd'))} {marker}",
            transform=ax.transAxes, fontsize=9, color=color,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            ha="left", va="top")


def main():
    args = parse_args()

    rwf = build_rwf_restormer(args.rwf_yaml, args.rwf_weights)
    seq = build_vanilla_restormer(args.seq_weights)
    prototypes = torch.load(args.prototypes, map_location="cpu")["prototypes"]
    extractor = BottleneckExtractor(rwf)

    rows = []
    for degradation, path in DOMAIN_TO_VAL_DIR.items():
        d = int(degradation.split("_")[0][-1])
        blur = load_and_crop(
            os.path.join(args.data_root, path, "input_crops", args.filename), args.crop_size)
        gt = load_and_crop(
            os.path.join(args.data_root, path, "target_crops", args.filename), args.crop_size)

        seq_out = infer(seq, blur)

        x_pad, _, _ = pad_to_factor(to_tensor(blur))
        emb = extractor.extract(x_pad)
        pred_key = route(emb, prototypes)
        pred_key = int(pred_key.split("_")[0][-1])
        adapter_id = pred_key - 2
        rwf_out = infer(rwf, blur, adapter_id=adapter_id)

        seq_psnr = compute_psnr(seq_out, gt)
        rwf_psnr = compute_psnr(rwf_out, gt)
        correct = pred_key == d

        print(f"D{d}: Seq = {seq_psnr:6.2f}dB | "
              f"RwF(->d{pred_key} {'✓' if correct else 'X'}) = {rwf_psnr:6.2f}dB")

        rows.append({
            "d": d, "blur": blur, "gt": gt,
            "seq": seq_out, "seq_psnr": seq_psnr,
            "rwf": rwf_out, "rwf_psnr": rwf_psnr,
            "pred_key": pred_key, "correct": correct,
        })

    extractor.close()

    print("\nRendering figure...")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    _, axes = plt.subplots(3, 6, figsize=(15, 6.6))
    row_labels = ["Input", "Sequential FT", "RwF (Ours)"]
    col_titles = [f"D{d+1}" for d in range(len(DOMAIN_TO_VAL_DIR))] + ["GT"]

    for r in range(3):
        for c in range(6):
            ax = axes[r, c]
            if c < 5:
                if r == 0:
                    ax.imshow(rows[c]["blur"])
                elif r == 1:
                    ax.imshow(rows[c]["seq"])
                    annotate_psnr(ax, rows[c]["seq_psnr"])
                else:
                    ax.imshow(rows[c]["rwf"])
                    annotate_psnr(ax, rows[c]["rwf_psnr"])
                    annotate_routing(ax, rows[c]["pred_key"], rows[c]["correct"])
            else:
                ax.imshow(rows[0]["gt"])

            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)

        axes[r, 0].set_ylabel(row_labels[r], fontsize=10, rotation=90, labelpad=8)

    plt.subplots_adjust(wspace=0.04, hspace=0.06)
    plt.savefig(args.output, format="pdf", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
