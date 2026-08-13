## This script extracts patches from the shift-variant blur dataset for training Restormer
## Source dataset structure:
##   /datasets/shift_variant_blur/variant_X/train/blur/
##   /datasets/shift_variant_blur/variant_X/train/sharp/
##   /datasets/shift_variant_blur/variant_X/val/blur/
##   /datasets/shift_variant_blur/variant_X/val/sharp/
##
## Output structure:
##   Motion_Deblurring/Datasets/train/ShiftVariant_V{X}/input_crops/
##   Motion_Deblurring/Datasets/train/ShiftVariant_V{X}/target_crops/
##   Motion_Deblurring/Datasets/val/ShiftVariant_V{X}/input_crops/
##   Motion_Deblurring/Datasets/val/ShiftVariant_V{X}/target_crops/

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def extract_patches(lr_img, hr_img, patch_size, overlap, p_max=0):
    """
    Extracts overlapping patches from LR/HR image pair

    Args:
        lr_img: Low-quality (blurred) image
        hr_img: High-quality (sharp/ground truth) image
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


def prepare_dataset(args):
    # source directories
    src_train_blur = os.path.join(args.src_root, f'variant_{args.variant}', 'train', 'blur')
    src_train_sharp = os.path.join(args.src_root, f'variant_{args.variant}', 'train', 'sharp')
    src_val_blur = os.path.join(args.src_root, f'variant_{args.variant}', 'val', 'blur')
    src_val_sharp = os.path.join(args.src_root, f'variant_{args.variant}', 'val', 'sharp')

    # target directories
    dataset_name = f'ShiftVariant_V{args.variant}'
    tar_train = os.path.join(args.tar_root, 'train', dataset_name)
    tar_val = os.path.join(args.tar_root, 'val', dataset_name)

    train_input_dir = os.path.join(tar_train, 'input_crops')
    train_target_dir = os.path.join(tar_train, 'target_crops')
    val_input_dir = os.path.join(tar_val, 'input_crops')
    val_target_dir = os.path.join(tar_val, 'target_crops')

    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_input_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    print(f"Preparing Shift-Variant Blur Dataset - Variant {args.variant}")
    print(f"\nSource paths:")
    print(f"Train blur directory: {src_train_blur}")
    print(f"Train sharp directory: {src_train_sharp}")
    print(f"Val blur directory: {src_val_blur}")
    print(f"Val sharp directory: {src_val_sharp}")
    print(f"\nTarget paths:")
    print(f"Train input directory: {train_input_dir}")
    print(f"Train target directory: {train_target_dir}")
    print(f"Val input directory: {val_input_dir}")
    print(f"Val target directory: {val_target_dir}")

    # verifying that source directories exist
    for path in [src_train_blur, src_train_sharp, src_val_blur, src_val_sharp]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source directory not found: {path}")

    # generating training file lists
    print("\nProcessing training images...")
    lr_files = natsorted(glob(os.path.join(src_train_blur, '*.png')))
    hr_files = natsorted(glob(os.path.join(src_train_sharp, '*.png')))

    print(f"Found {len(lr_files)} blur images")
    print(f"Found {len(hr_files)} sharp images")

    if len(lr_files) != len(hr_files):
        print("Warning: Mismatch in number of blur/sharp images!")

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
            ) for file_pair in tqdm(train_files, desc="Training")
        )
        total_patches = sum(results)
    else:
        total_patches = 0
        for file_pair in tqdm(train_files, desc="Training"):
            num = process_train_image(
                file_pair, args.patch_size, args.overlap,
                train_input_dir, train_target_dir, args.p_max
            )
            total_patches += num

    print(f"Generated {total_patches} training patches")

    # generating validation file lists
    print("\nProcessing validation images...")
    lr_files = natsorted(glob(os.path.join(src_val_blur, '*.png')))
    hr_files = natsorted(glob(os.path.join(src_val_sharp, '*.png')))

    print(f"Found {len(lr_files)} blur images")
    print(f"Found {len(hr_files)} sharp images")

    val_files = list(zip(lr_files, hr_files))

    if args.num_workers > 1:
        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_val_image)(
                file_pair, args.val_patch_size, 
                val_input_dir, val_target_dir, not args.no_center_crop
            ) for file_pair in tqdm(val_files, desc="Validation")
        )
        val_count = sum(results)
    else:
        val_count = 0
        for file_pair in tqdm(val_files, desc="Validation"):
            if process_val_image(
                file_pair, args.val_patch_size,
                val_input_dir, val_target_dir, not args.no_center_crop
            ):
                val_count += 1

    print(f"Processed {val_count} validation images")


def prepare_validation_dataset_only(args):
    # source directories
    src_val_blur = os.path.join(args.src_root, f'variant_{args.variant}', 'val', 'blur')
    src_val_sharp = os.path.join(args.src_root, f'variant_{args.variant}', 'val', 'sharp')

    # target directories
    dataset_name = f'ShiftVariant_V{args.variant}_Full_Images'
    tar_val = os.path.join(args.tar_root, 'val', dataset_name)

    val_input_dir = os.path.join(tar_val, 'input_crops')
    val_target_dir = os.path.join(tar_val, 'target_crops')

    os.makedirs(val_input_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    print(f"Preparing Validation Set Only - Variant {args.variant}")
    print(f"\nSource paths:")
    print(f"Val blur directory: {src_val_blur}")
    print(f"Val sharp directory: {src_val_sharp}")
    print(f"\nTarget paths:")
    print(f"Val input directory: {val_input_dir}")
    print(f"Val target directory: {val_target_dir}")

     # verifying that source directories exist
    for path in [src_val_blur, src_val_sharp]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source directory not found: {path}")

    # generating validation file lists
    print("\nProcessing validation images...")
    lr_files = natsorted(glob(os.path.join(src_val_blur, '*.png')))
    hr_files = natsorted(glob(os.path.join(src_val_sharp, '*.png')))

    print(f"Found {len(lr_files)} blur images")
    print(f"Found {len(hr_files)} sharp images")

    val_files = list(zip(lr_files, hr_files))

    if args.num_workers > 1:
        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_val_image)(
                file_pair, args.val_patch_size, 
                val_input_dir, val_target_dir, not args.no_center_crop
            ) for file_pair in tqdm(val_files, desc="Validation")
        )
        val_count = sum(results)
    else:
        val_count = 0
        for file_pair in tqdm(val_files, desc="Validation"):
            if process_val_image(
                file_pair, args.val_patch_size,
                val_input_dir, val_target_dir, not args.no_center_crop
            ):
                val_count += 1

    print(f"Processed {val_count} validation images")


if __name__ == '__main__':
    """
    Example Usages:
        # Prepare variant 1 with default settings
        python generate_patches_shift_variant.py --variant 1

        # Prepare variant 2 with custom paths
        python generate_patches_shift_variant.py --variant 2 \\
            --src-root /path/to/datasets/shift_variant_blur \\
            --tar-root Datasets

        # Prepare with different patch sizes
        python generate_patches_shift_variant.py --variant 1 \\
            --patch-size 256 --overlap 128 --val-patch-size 256

        # Prepare only validation set (keep full images)
        python generate_patches_shift_variant.py --variant 1 --prepare_val_only \\
            --no-center-crop
    """

    parser = argparse.ArgumentParser(
        description='Prepares shift-variant blur dataset for Restormer training',
        formatter_class=argparse.RawDescriptionHelpFormatter
        )

    parser.add_argument('--variant', type=int, required=True, 
                        choices=[1, 2, 3, 4, 5, 6], help='Variant must be between 1 and 6')
    parser.add_argument('--src-root', type=str, default='../../datasets/shift_variant_blur',
                        help='Root directory of source dataset')
    parser.add_argument('--tar-root', type=str, default='Datasets',
                        help='Root directory for output dataset')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Training patch size (default: 512)')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between patches (default: 256)')
    parser.add_argument('--val-patch-size', type=int, default=256,
                        help='Validation patch size for center crop (default: 256)')
    parser.add_argument('--p-max', type=int, default=0,
                        help='Minimum image size threshold (default: 0, always extract patches)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--no-center-crop', action='store_true',
                        help='Disable center cropping for validation (keep full images)')
    parser.add_argument('--prepare_val_only', action='store_true',
                        help='Only prepare validation set (skip training patches)')

    args = parser.parse_args()

    if args.prepare_val_only:
        prepare_validation_dataset_only(args)
    else:
        prepare_dataset(args)
