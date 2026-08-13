"""
Extracts backbone-only (adapter_id=-1) bottleneck embeddings from images across 
multiple domains and projects them to 2D via UMAP for separability analysis.

-> Do real motion blurs form distinct natural clusters in the latent space? 
-> Are the synthetic domains a synthetic artifact?

Default plot: 9 clusters in one panel
Synthetic shift-variant blur: V1, V2, V3, V4, V5, V6
Real motion blur: REDS, RealBlur-J, RealBlur-R

Example Usage:
    CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py \\
        --yaml_file ./Options/ShiftVariant_V1_thru_V6_Restormer_Adapters.yml \\
        --weights ../experiments/archived_checkpoints/V1_thru_V6_Adapters/net_g_latest.pth \\
        --samples_per_domain 100 \\
        --output_npz extracted_embeddings.npz
"""

import os
import sys
import random
import argparse
from glob import glob

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from natsort import natsorted
from tqdm import tqdm


DOMAIN_CONFIG = [
    ("V1", "./Datasets/val/ShiftVariant_V1_Full_Images/input_crops"),
    ("V2", "./Datasets/val/ShiftVariant_V2_Full_Images/input_crops"),
    ("V3", "./Datasets/val/ShiftVariant_V3_Full_Images/input_crops"),
    ("V4", "./Datasets/val/ShiftVariant_V4_Full_Images/input_crops"),
    ("V5", "./Datasets/val/ShiftVariant_V5_Full_Images/input_crops"),
    ("V6", "./Datasets/val/ShiftVariant_V6_Full_Images/input_crops"),
    ("REDS", "./Datasets/val/REDS_Full_Images/input_crops"),
    ("RealBlur-J", "./Datasets/val/RealBlur_J_Full_Images/input_crops"),
    ("RealBlur-R", "./Datasets/val/RealBlur_R_Full_Images/input_crops"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml_file", required=True,
                   help="ConDA-Restormer YAML used to build the network")
    p.add_argument("--weights", required=True, 
                   help="ConDA-Restormer checkpoint (Either V1-only or V1→V6)")
    p.add_argument("--samples_per_domain", type=int, default=100)
    p.add_argument("--output_npz", 
                   default="../experiments/archived_checkpoints/extracted_embeddings/extracted_embeddings.npz")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()


def build_conda(yaml_file, weights):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_domain_incremental import build_model

    class _A:
        pass

    a = _A()
    a.yaml_file = yaml_file
    a.weights = weights
    return build_model(a).cuda().eval()


class BottleneckExtractor:
    """
    L2-normalized GAP'd bottleneck features using adapter_id=-1
    so embeddings exist in the shared backbone-only feature space.
    """
    def __init__(self, model):
        self.model = model
        self._features = {}
        self._handle = model.latent[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._features["bottleneck"] = output

    def extract_embedding(self, img_tensor):
        with torch.no_grad():
            _ = self.model(img_tensor, adapter_id=-1)  # backbone only

        bottleneck = self._features["bottleneck"]
        embedding = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)  # [1, C]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def close(self):
        self._handle.remove()


def sample_files(directory, n, rng):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(glob(os.path.join(directory, "**", ext), recursive=True))
    files = natsorted(files)
    if not files:
        return []
    if len(files) <= n:
        return files
    indices = sorted(rng.sample(range(len(files)), n))
    return [files[i] for i in indices]


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print("Loading ConDA-Restormer backbone...")
    model = build_conda(args.yaml_file, args.weights)
    extractor = BottleneckExtractor(model)

    all_embeddings, all_labels, kept_names = [], [], []

    for _, (name, directory) in enumerate(DOMAIN_CONFIG):
        if not os.path.isdir(directory):
            print(f"[SKIP] {name}: directory not found ({directory})")
            continue

        files = sample_files(directory, args.samples_per_domain, rng)
        if not files:
            print(f"[SKIP] {name}: no images in {directory}")
            continue

        print(f"\n[{name}] {len(files)} images from {directory}")
        embeds = []
        for f in tqdm(files, desc=name, leave=False):
            img = np.float32(load_img(f)) / 255.0
            padding_factor = 8
            img_tensor = to_tensor(img)
            _, _, h, w = img_tensor.shape
            pad_h = (padding_factor - h % padding_factor) % padding_factor
            pad_w = (padding_factor - w % padding_factor) % padding_factor
            if pad_h > 0 or pad_w > 0:
                img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), "reflect")
            e = extractor.extract_embedding(img_tensor).cpu().numpy()  # [1, C]
            embeds.append(e[0])

        embeds = np.stack(embeds, axis=0)
        all_embeddings.append(embeds)
        all_labels.append(np.full(embeds.shape[0], len(kept_names), dtype=np.int32))
        kept_names.append(name)
        print(f"-> {embeds.shape[0]} embeddings of dim {embeds.shape[1]}")

    extractor.close()

    if not all_embeddings:
        sys.exit("No embeddings extracted; check DOMAIN_CONFIG paths.")

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\nTotal: {embeddings.shape[0]} embeddings across "
          f"{len(kept_names)} domains, dim={embeddings.shape[1]}")

    np.savez(args.output_npz, embeddings=embeddings, labels=labels, 
             domain_names=np.array(kept_names), seed=args.seed)
    print(f"Raw data saved: {args.output_npz}")
