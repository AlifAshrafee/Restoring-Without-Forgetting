"""
Generate cross-task domain-incremental image restoration benchmark from DIV2K.

Domain sequence:
    D1: Gaussian Denoise (pre-trained backbone) -> D2: Deblur -> D3: Derain -> D4: Dehaze -> D5: Lowlight

Usage:
    python generate_degradation_domains.py \
        --div2k_dir /path/to/DIV2K \
        --output_dir /path/to/output \
        --seed 42

Expected DIV2K directory structure:
    DIV2K/
    ├── DIV2K_train_HR/       # 800 images (0001.png - 0800.png)
    └── DIV2K_valid_HR/       # 100 images (0801.png - 0900.png)

Output structure:
    output_dir/
    ├── D1_denoise/
    │   ├── train/
    │   │   ├── degraded/     # 800 noisy images
    │   │   └── gt/           # 800 clean images
    │   └── val/
    │       ├── degraded/     # 100 noisy images
    │       └── gt/           # 100 clean images
    ├── D2_deblur/
    │   └── ...
    ├── D3_derain/
    │   └── ...
    ├── D4_dehaze/
    │   └── ...
    ├── D5_lowlight/
    │   └── ...
    └── metadata.json
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm

from degradations import add_gaussian_noise, add_motion_blur, add_haze, add_rain, add_low_light


DOMAIN_FUNC_MAP = {
    "D1_denoise": add_gaussian_noise,
    "D2_deblur": add_motion_blur,
    "D3_derain": add_rain,
    "D4_dehaze": add_haze,
    "D5_lowlight": add_low_light,
}


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def process_single_image(src_path, degraded_path, gt_path, 
                         domain_key, img_seed):
    np.random.seed(img_seed)
    func = DOMAIN_FUNC_MAP[domain_key]
    img = load_image(src_path)
    degraded = func(img)
    save_image(degraded, degraded_path)
    save_image(img, gt_path)

    meta = {
        "source": os.path.basename(src_path),
        "seed": img_seed,
    }

    return meta


def process_domain(domain_key, src_dir, output_dir, split, 
                   global_seed, num_workers=8):
    src_files = sorted([f for f in os.listdir(src_dir)])
    if not src_files:
        raise RuntimeError(f"No images found in {src_dir}")

    degraded_dir = os.path.join(output_dir, domain_key, split, "degraded")
    gt_dir = os.path.join(output_dir, domain_key, split, "gt")
    os.makedirs(degraded_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # domain-specific seed offset so different domains get different randomness
    domain_idx = list(DOMAIN_FUNC_MAP.keys()).index(domain_key)
    split_offset = 0 if split == "train" else 100000

    print(f"\n{'-'*60}")
    print(f"Generating {domain_key} [{split}]: {len(src_files)} images")
    print(f"{'-'*60}")

    # Haze requires MiDaS which uses GPU
    if domain_key == "D4_dehaze":
        num_workers = 1  # force single-worker for haze to avoid loading multiple MiDaS instances

    all_meta = []

    if num_workers <= 1:
        for i, fname in enumerate(tqdm(src_files, desc=domain_key)):
            src_path = os.path.join(src_dir, fname)
            out_name = os.path.splitext(fname)[0] + ".png"
            degraded_path = os.path.join(degraded_dir, out_name)
            gt_path = os.path.join(gt_dir, out_name)

            img_seed = global_seed + domain_idx * 10000 + split_offset + i

            meta = process_single_image(src_path, degraded_path, gt_path, 
                                        domain_key, img_seed)
            all_meta.append(meta)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, fname in enumerate(src_files):
                src_path = os.path.join(src_dir, fname)
                out_name = os.path.splitext(fname)[0] + ".png"
                degraded_path = os.path.join(degraded_dir, out_name)
                gt_path = os.path.join(gt_dir, out_name)

                img_seed = global_seed + domain_idx * 10000 + split_offset + i

                fut = executor.submit(process_single_image, src_path, degraded_path, 
                                      gt_path, domain_key, img_seed)
                futures[fut] = fname

            for fut in tqdm(as_completed(futures), total=len(futures), desc=domain_key):
                meta = fut.result()
                all_meta.append(meta)

    return {
        "domain": domain_key,
        "split": split,
        "num_images": len(src_files),
        "source_dir": src_dir,
        "images": sorted(all_meta, key=lambda m: m["source"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate domain-incremental restoration benchmark from DIV2K."
    )
    parser.add_argument("--div2k_dir", type=str, default="datasets/DIV2K",
                        help="Path to DIV2K root (containing DIV2K_train_HR and DIV2K_valid_HR)")
    parser.add_argument("--output_dir", type=str, default="datasets/multiple_degradations", 
                        help="Output directory for generated benchmark")
    parser.add_argument("--domains", type=str, nargs="*", default=None, 
                        help="Domains to generate (e.g., D3_derain). Default: all domains")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val"], 
                        choices=["train", "val"], help="Which split(s) to generate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Global random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of parallel workers")
    args = parser.parse_args()

    div2k_root = Path(args.div2k_dir)
    split_dirs = {
        "train": div2k_root / "DIV2K_train_HR",
        "val": div2k_root / "DIV2K_valid_HR",
    }

    for split in args.splits:
        if not split_dirs[split].is_dir():
            print(f"ERROR: {split_dirs[split]} does not exist.")
            print("Expected DIV2K structure:")
            print("DIV2K/")
            print("├── DIV2K_train_HR/  (800 images)")
            print("└── DIV2K_valid_HR/  (100 images)")
            sys.exit(1)

    domains = args.domains or list(DOMAIN_FUNC_MAP.keys())
    for d in domains:
        if d not in DOMAIN_FUNC_MAP:
            print(f"ERROR: Unknown domain: {d}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    all_metadata = {
        "benchmark": "DIV2K-CrossTask-4Domain-Incremental",
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain_order": list(DOMAIN_FUNC_MAP.keys()),
        "domains": {},
    }

    for domain_key in domains:
        all_metadata["domains"][domain_key] = {}
        workers = args.num_workers

        for split in args.splits:
            meta = process_domain(domain_key=domain_key, src_dir=str(split_dirs[split]), 
                                  output_dir=args.output_dir, split=split, 
                                  global_seed=args.seed, num_workers=workers)
            all_metadata["domains"][domain_key][split] = {
                "num_images": meta["num_images"]
            }

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'-'*60}")
    print("BENCHMARK GENERATION COMPLETE")
    print(f"{'-'*60}")
    print(f"Output: {args.output_dir}")
    print(f"Metadata: {meta_path}")
    print(f"\nDomain order ({len(DOMAIN_FUNC_MAP)} stages):")
    for i, d in enumerate(DOMAIN_FUNC_MAP.keys(), 1):
        print(f"Stage {i}: {d}")


if __name__ == "__main__":
    main()