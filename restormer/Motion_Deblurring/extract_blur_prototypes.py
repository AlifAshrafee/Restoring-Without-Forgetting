# ------------------------------------------------------------------------
# Restormer Adapters - Blur Prototype Extraction Script
# Usage:
#   PYTHONPATH=.. python extract_blur_prototypes.py \
#       --weights ../experiments/archived_checkpoints/V1_thru_V6/net_g_latest.pth \
#       --yaml_file Options/Shift_Variant_V1_thru_V6_Deblurring_Restormer_Adapters.yml \
#       --data_root ../Motion_Deblurring/Datasets/train \
#       --output prototypes.pth \
#       --max_samples 13800 (50% of all patches per domain)
# ------------------------------------------------------------------------

import os
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm

import torch
import torch.nn.functional as F
import cv2
import yaml

from basicsr.models.archs.restormer_adapters_arch import RestormerAdapters


def load_config(yaml_path):
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=Loader)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def build_model(args):
    config = load_config(args.yaml_file)
    network_config = config["network_g"].copy()
    network_config.pop("type", None)
    adapter_config = network_config.pop("adapter_config", None)

    model = RestormerAdapters(**network_config, adapter_config=adapter_config)

    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)

    model.prepare_adapter_list_for_loading(state_dict)
    model.load_state_dict(state_dict, strict=False)

    num_committed = len(model.adapter_list)
    print(f"Loaded checkpoint from: {args.weights}")
    print(f"Committed adapters: {num_committed}")
    print(f"Total adapter sets: {num_committed + 1}")

    return model


class BottleneckExtractor:
    """
    Extracts bottleneck features from RestormerAdapters using a forward hook
    on the last TransformerBlock in self.latent.
    Always uses adapter_id=-1 (backbone only) to ensure all prototypes
    are in the same feature space.
    """
    def __init__(self, model):
        self.model = model
        self._features = {}
        self._handle = model.latent[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._features["bottleneck"] = output

    def extract(self, img_tensor):
        with torch.no_grad():
            _ = self.model(img_tensor, adapter_id=-1)  # backbone only

        bottleneck = self._features["bottleneck"]
        embedding = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)  # [B, C]
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def close(self):
        self._handle.remove()


def extract_prototypes(args):
    model = build_model(args)
    model.cuda()
    model.eval()

    extractor = BottleneckExtractor(model)

    num_variants = args.num_variants
    print(f"\nExtracting prototypes for {num_variants} variants (V1 through V{num_variants})")
    print(f"Feature extraction: backbone only (adapter_id=-1) for ALL variants")

    factor = 8  # padding factor for Restormer

    prototypes = {}
    metadata = {}

    for variant in range(1, num_variants + 1):
        data_dir = os.path.join(args.data_root, f"ShiftVariant_V{variant}", "input_crops")
        if not os.path.exists(data_dir):
            print(f"[WARNING] Data directory not found for V{variant}: {data_dir}, skipping.")
            continue

        files = natsorted(glob(os.path.join(data_dir, "*.png")))
        if len(files) == 0:
            print(f"\n[WARNING] No PNG images found in {data_dir}, skipping.")
            continue

        if args.max_samples is not None and len(files) > args.max_samples:
            rng = np.random.RandomState(seed=42)
            indices = rng.choice(len(files), args.max_samples, replace=False)
            indices.sort()
            files = [files[i] for i in indices]

        print(f"\nV{variant}: processing {len(files)} images from {data_dir}")

        embedding_sum = None
        count = 0

        for file_ in tqdm(files, desc=f"V{variant}", unit="img"):
            img = np.float32(load_img(file_)) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

            _, _, h, w = img_tensor.shape
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            if pad_h > 0 or pad_w > 0:
                img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), "reflect")

            embedding = extractor.extract(img_tensor)  # [1, C], L2-normalized

            if embedding_sum is None:
                embedding_sum = embedding.cpu().squeeze(0)
            else:
                embedding_sum += embedding.cpu().squeeze(0)
            count += 1

            if count % 100 == 0:
                torch.cuda.empty_cache()

        prototype = F.normalize(embedding_sum / count, p=2, dim=0)  # [C]

        prototypes[variant] = prototype
        metadata[variant] = {
            "variant": variant,
            "num_samples": count,
            "data_dir": data_dir,
            "embedding_dim": prototype.shape[0],
        }

        print(f"Prototype V{variant}: dim={prototype.shape[0]}, computed from {count} samples")

    extractor.close()

    save_dict = {
        "prototypes": prototypes,
        "metadata": metadata,
        "config": {
            "yaml_file": args.yaml_file,
            "weights": args.weights,
            "max_samples": args.max_samples,
            "extraction_point": "bottleneck_after_latent_blocks_gap",
            "normalization": "L2",
            "adapter_id": -1,  # backbone only for all variants
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(save_dict, args.output)
    print(f"\nPrototypes saved to: {args.output}")
    print(f"Variants extracted: {sorted(prototypes.keys())}")
    print(f"Embedding dimension: {next(iter(prototypes.values())).shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract blur domain prototypes from RestormerAdapters"
    )
    parser.add_argument(
        "--weights", type=str,
        default="../experiments/archived_checkpoints/V1_V2_V3_V4_V5_ft_V6_Adapters/net_g_latest.pth",
        help="Path to RestormerAdapters checkpoint with all trained adapters",
    )
    parser.add_argument(
        "--yaml_file", type=str, 
        default="Options/Shift_Variant_V1_V2_V3_V4_V5_ft_V6_Deblurring_Restormer_Adapters.yml",
        help="Path to YAML config for RestormerAdapters",
    )
    parser.add_argument(
        "--data_root", type=str, 
        default="Datasets/train",
        help="Root directory containing train/ShiftVariant_V{k}/ subdirectories",
    )
    parser.add_argument(
        "--output", type=str, 
        default="../experiments/archived_checkpoints/prototypes/blur_prototypes.pth",
        help="Output path for the prototypes .pth file",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max training samples used for per variant prototype calculation (None = use all)",
    )
    parser.add_argument(
        "--num_variants", type=int, default=6,
        help="Number of variants to extract prototypes for",
    )

    args = parser.parse_args()
    extract_prototypes(args)
