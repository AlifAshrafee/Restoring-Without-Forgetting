## This script extracts patches from the cross-degradation benchmark
##
## Source dataset structure:
##   datasets/multiple_degradations/D2_deblur/train/degraded/
##   datasets/multiple_degradations/D2_deblur/train/gt/
##   datasets/multiple_degradations/D2_deblur/val/degraded/
##   datasets/multiple_degradations/D2_deblur/val/gt/
##   ... (same for D3_derain, D4_dehaze, D5_lowlight)
##
## Output structure:
##   Datasets/train/D2_deblur/input_crops/
##   Datasets/train/D2_deblur/target_crops/
##   Datasets/val/D2_deblur/input_crops/
##   Datasets/val/D2_deblur/target_crops/
##   ... (same for D3_derain, D4_dehaze, D5_lowlight)

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


VALID_DOMAINS = ["D1_denoise", "D2_deblur", "D3_derain", "D4_dehaze", "D5_lowlight"]


def extract_patches(lr_img, hr_img, patch_size, overlap, p_max=0):
    """
    Extracts overlapping patches from degraded/GT image pair.

    Args:
        lr_img: Degraded image
        hr_img: Ground truth (clean) image
        patch_size: Size of patches to extract
        overlap: Overlap between adjacent patches
        p_max: Minimum image dimension threshold (0 = always extract patches)

    Returns:
        List of (lr_patch, hr_patch) tuples
    """
    patches = []
    h, w = lr_img.shape[:2]

    if h > p_max and w > p_max:
        # generating patch start positions with overlap
        h_positions = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=int))
        w_positions = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=int))

        # ensuring coverage of the full image by adding edge positions
        if h - patch_size not in h_positions:
            h_positions.append(h - patch_size)
        if w - patch_size not in w_positions:
            w_positions.append(w - patch_size)

        for i in h_positions:
            for j in w_positions:
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size, :]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size, :]
                patches.append((lr_patch, hr_patch))
    else:
        # if image smaller than patch size, use as-is
        patches.append((lr_img, hr_img))

    return patches


def process_train_image(file_pair, patch_size, overlap, lr_tar, hr_tar, p_max=0):
    """Processes a single training image pair by extracting patches."""
    lr_file, hr_file = file_pair
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]

    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)

    if lr_img is None:
        print(f"Warning: Could not read {lr_file}")
        return 0
    if hr_img is None:
        print(f"Warning: Could not read {hr_file}")
        return 0

    patches = extract_patches(lr_img, hr_img, patch_size, overlap, p_max)

    for idx, (lr_patch, hr_patch) in enumerate(patches, 1):
        if len(patches) > 1:
            lr_savename = os.path.join(lr_tar, f'{filename}-{idx}.png')
            hr_savename = os.path.join(hr_tar, f'{filename}-{idx}.png')
        else:
            lr_savename = os.path.join(lr_tar, f'{filename}.png')
            hr_savename = os.path.join(hr_tar, f'{filename}.png')

        cv2.imwrite(lr_savename, lr_patch)
        cv2.imwrite(hr_savename, hr_patch)

    return len(patches)


def process_val_image(file_pair, val_patch_size, lr_tar, hr_tar, center_crop=True):
    """Processes a single validation image pair"""
    lr_file, hr_file = file_pair
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]

    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)

    if lr_img is None:
        print(f"Warning: Could not read {lr_file}")
        return False
    if hr_img is None:
        print(f"Warning: Could not read {hr_file}")
        return False

    h, w = lr_img.shape[:2]

    if center_crop and h >= val_patch_size and w >= val_patch_size:
        i = (h - val_patch_size) // 2
        j = (w - val_patch_size) // 2
        lr_img = lr_img[i:i+val_patch_size, j:j+val_patch_size, :]
        hr_img = hr_img[i:i+val_patch_size, j:j+val_patch_size, :]

    lr_savename = os.path.join(lr_tar, f'{filename}.png')
    hr_savename = os.path.join(hr_tar, f'{filename}.png')

    cv2.imwrite(lr_savename, lr_img)
    cv2.imwrite(hr_savename, hr_img)

    return True


def prepare_domain(args, domain):
    # source directories
    src_train_degraded = os.path.join(args.src_root, domain, 'train', 'degraded')
    src_train_gt = os.path.join(args.src_root, domain, 'train', 'gt')
    src_val_degraded = os.path.join(args.src_root, domain, 'val', 'degraded')
    src_val_gt = os.path.join(args.src_root, domain, 'val', 'gt')

    # target directories
    tar_train = os.path.join(args.tar_root, 'train', domain)
    tar_val = os.path.join(args.tar_root, 'val', domain)

    train_input_dir = os.path.join(tar_train, 'input_crops')
    train_target_dir = os.path.join(tar_train, 'target_crops')
    val_input_dir = os.path.join(tar_val, 'input_crops')
    val_target_dir = os.path.join(tar_val, 'target_crops')

    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_input_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    print(f"\n{'-'*60}")
    print(f"Preparing {domain}")
    print(f"{'-'*60}")
    print(f"Source:  {os.path.join(args.src_root, domain)}")
    print(f"Target:  {os.path.join(args.tar_root, '{{train,val}}', domain)}")

    # verifying that source directories exist
    for path in [src_train_degraded, src_train_gt, src_val_degraded, src_val_gt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source directory not found: {path}")

    # --- Training patches ---
    print(f"\nProcessing training images...")
    lr_files = natsorted(glob(os.path.join(src_train_degraded, '*.png')))
    hr_files = natsorted(glob(os.path.join(src_train_gt, '*.png')))

    print(f"Found {len(lr_files)} degraded / {len(hr_files)} GT images")

    if len(lr_files) != len(hr_files):
        print(f"Warning: Mismatch in number of degraded/GT images!")

    # verifying filename matching
    for lr, hr in zip(lr_files[:3], hr_files[:3]):
        lr_name = os.path.basename(lr)
        hr_name = os.path.basename(hr)
        if lr_name != hr_name:
            print(f"Warning: Filename mismatch - {lr_name} vs {hr_name}")

    train_files = list(zip(lr_files, hr_files))

    # processing with parallel workers
    if args.num_workers > 1:
        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_train_image)(
                file_pair, args.patch_size, args.overlap, 
                train_input_dir, train_target_dir, args.p_max
            ) for file_pair in tqdm(train_files, desc=f"{domain} train")
        )
        total_patches = sum(results)
    else:
        total_patches = 0
        for file_pair in tqdm(train_files, desc=f"{domain} train"):
            num = process_train_image(
                file_pair, args.patch_size, args.overlap,
                train_input_dir, train_target_dir, args.p_max
            )
            total_patches += num

    print(f"Generated {total_patches} training patches")

    # --- Validation images ---
    print(f"\nProcessing validation images...")
    lr_files = natsorted(glob(os.path.join(src_val_degraded, '*.png')))
    hr_files = natsorted(glob(os.path.join(src_val_gt, '*.png')))

    print(f"Found {len(lr_files)} degraded / {len(hr_files)} GT images")

    val_files = list(zip(lr_files, hr_files))

    if args.num_workers > 1:
        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_val_image)(
                file_pair, args.val_patch_size, 
                val_input_dir, val_target_dir, not args.no_center_crop
            ) for file_pair in tqdm(val_files, desc=f"{domain} val")
        )
        val_count = sum(results)
    else:
        val_count = 0
        for file_pair in tqdm(val_files, desc=f"{domain} val"):
            if process_val_image(
                file_pair, args.val_patch_size,
                val_input_dir, val_target_dir, not args.no_center_crop
            ):
                val_count += 1

    print(f"Processed {val_count} validation images")

    return total_patches, val_count

def prepare_val_only(args, domain):
    # source directories
    src_val_degraded = os.path.join(args.src_root, domain, 'val', 'degraded')
    src_val_gt = os.path.join(args.src_root, domain, 'val', 'gt')

    # target directories
    dataset_name = f'{domain}_Full_Images'
    tar_val = os.path.join(args.tar_root, 'val', dataset_name)

    val_input_dir = os.path.join(tar_val, 'input_crops')
    val_target_dir = os.path.join(tar_val, 'target_crops')

    os.makedirs(val_input_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    print(f"\n{'-'*60}")
    print(f"Preparing Validation Only - {domain}")
    print(f"{'-'*60}")
    print(f"Source: {src_val_degraded}")
    print(f"Target: {tar_val}")

     # verifying that source directories exist
    for path in [src_val_degraded, src_val_gt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source directory not found: {path}")

    # generating validation file lists
    print("\nProcessing validation images...")
    lr_files = natsorted(glob(os.path.join(src_val_degraded, '*.png')))
    hr_files = natsorted(glob(os.path.join(src_val_gt, '*.png')))

    print(f"  Found {len(lr_files)} degraded / {len(hr_files)} GT images")

    val_files = list(zip(lr_files, hr_files))

    if args.num_workers > 1:
        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_val_image)(
                file_pair, args.val_patch_size,
                val_input_dir, val_target_dir, not args.no_center_crop
            ) for file_pair in tqdm(val_files, desc=f"{domain} val")
        )
        val_count = sum(results)
    else:
        val_count = 0
        for file_pair in tqdm(val_files, desc=f"{domain} val"):
            if process_val_image(
                file_pair, args.val_patch_size,
                val_input_dir, val_target_dir, not args.no_center_crop
            ):
                val_count += 1

    print(f"Processed {val_count} validation images")
    return val_count


if __name__ == '__main__':
    """
    Example Usages:
        # Prepare all 4 degradation domains with default settings
        python generate_patches_degradations.py

        # Prepare a single domain
        python generate_patches_degradations.py --domains D2_deblur

        # Prepare two specific domains
        python generate_patches_degradations.py --domains D3_derain D5_lowlight

        # Custom paths and patch sizes
        python generate_patches_degradations.py \\
            --src-root /path/to/datasets/multiple_degradations \\
            --tar-root Datasets \\
            --patch-size 256 --overlap 128

        # Prepare only validation set (full images, no center crop)
        python generate_patches_degradations.py --prepare_val_only --no-center-crop
    """

    parser = argparse.ArgumentParser(
        description='Prepare cross-degradation benchmark patches for Restormer / ConDA training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--domains', type=str, nargs='*', default=None,
                        choices=VALID_DOMAINS,
                        help='Which domains to process (default: all four)')
    parser.add_argument('--src-root', type=str,
                        default='../../datasets/multiple_degradations',
                        help='Root directory of source degradation datasets')
    parser.add_argument('--tar-root', type=str, default='Datasets',
                        help='Root directory for output patches')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Training patch size (default: 512)')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between patches (default: 256)')
    parser.add_argument('--val-patch-size', type=int, default=256,
                        help='Validation patch size for center crop (default: 256)')
    parser.add_argument('--p-max', type=int, default=0,
                        help='Minimum image size threshold (default: 0)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--no-center-crop', action='store_true',
                        help='Disable center cropping for validation')
    parser.add_argument('--prepare_val_only', action='store_true',
                        help='Only prepare validation set (skip training patches)')

    args = parser.parse_args()

    domains = args.domains or VALID_DOMAINS

    print("Cross-Degradation Benchmark — Patch Generation")
    print(f"Domains: {', '.join(domains)}")
    print(f"Patch size: {args.patch_size}, Overlap: {args.overlap}")
    print(f"Source: {args.src_root}")
    print(f"Target: {args.tar_root}")

    summary = {}

    for domain in domains:
        if args.prepare_val_only:
            val_count = prepare_val_only(args, domain)
            summary[domain] = {"val": val_count}
        else:
            train_count, val_count = prepare_domain(args, domain)
            summary[domain] = {"train_patches": train_count, "val": val_count}

    print(f"\n{'-'*60}")
    print("SUMMARY")
    print(f"{'-'*60}")
    for domain, counts in summary.items():
        if "train_patches" in counts:
            print(f"{domain:<15s}  train: {counts['train_patches']:>6d} patches val: {counts['val']:>4d} images")
        else:
            print(f"{domain:<15s}  val: {counts['val']:>4d} images")
