"""
Unified Shift-Variant Blur Synthesis Pipeline

This script supports two modes:
1. Single image mode: Apply blur to one image with visualizations
2. Dataset mode: Generate blur-sharp paired datasets for training deblurring models

Usage:
# Single image with visualizations
python run_blur_synthesis_pipeline.py single \\
    --input image.png \\
    --library psf_library_variant_1.npz \\
    --visualize

# Generate dataset for one variant
python run_blur_synthesis_pipeline.py dataset \\
    --train-dir datasets/DIV2K_train_HR \\
    --val-dir datasets/DIV2K_valid_HR \\
    --variant 1 \\
    --output-root datasets/shift_variant_blur

# Generate datasets for all 6 variants
python run_blur_synthesis_pipeline.py dataset \\
    --train-dir datasets/DIV2K_train_HR \\
    --val-dir datasets/DIV2K_valid_HR \\
    --variant all \\
    --output-root datasets/shift_variant_blur

Output Folder Structure (Dataset Mode):
--------------------------------------
datasets/shift_variant_blur/
├── variant_1/                      # Domain/Task 1
│   ├── train/
│   │   ├── blur/                   # Blurred images (input to model)
│   │   │   ├── 0001.png
│   │   │   └── ...
│   │   └── sharp/                  # Ground truth (target)
│   │       ├── 0001.png
│   │       └── ...
│   ├── val/
│   │   ├── blur/
│   │   └── sharp/
│   └── metadata.json               # Variant config & stats
├── variant_2/                      # Domain/Task 2
│   └── ...
├── ...
├── variant_6/                      # Domain/Task 6
│   └── ...
└── dataset_info.json               # Global dataset metadata
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from psf_tools.library import load_psf_library
from psf_tools.convolution_visualization import visualize_field_assignment, visualize_comparison
from psf_tools.overlap_add_conv import overlap_add_convolution, compute_sensor_fov_from_library


DEFAULT_CONFIG = {
    'patch_size': 512,
    'psf_crop_size': 512,
    'overlap_ratio': 0.5,
    'window_type': 'hann',
    'num_variants': 6,
    'library_dir': 'star_psfs_separated',
    'supported_extensions': ['.png'],
}


def get_image_paths(directory):
    extensions = DEFAULT_CONFIG['supported_extensions']

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_paths.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))

    return sorted(list(set(image_paths)))


def load_image(path):
    image = plt.imread(path)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0

    if image.ndim == 2:  # grayscale
        image = np.stack([image, image, image], axis=-1)

    return image


def save_image(image, path):
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.clip(image, 0, 1)
    plt.imsave(path, image)


def create_metadata(variant, psf_library, config, stats):
    field_positions = psf_library['field_positions_degrees']

    return {
        'variant': variant,
        'created_at': datetime.now().isoformat(),
        'psf_library': {
            'num_fields': len(psf_library['field_numbers']),
            'field_extent_x': [float(field_positions[:, 0].min()), float(field_positions[:, 0].max())],
            'field_extent_y': [float(field_positions[:, 1].min()), float(field_positions[:, 1].max())],
            'psf_shape': list(psf_library['psf_kernels'][psf_library['field_numbers'][0]].shape),
        },
        'processing_config': {
            'patch_size': config['patch_size'],
            'psf_crop_size': config['psf_crop_size'],
            'overlap_ratio': config['overlap_ratio'],
            'sensor_fov_degrees': config['sensor_fov'],
            'window_type': config['window_type'],
        },
        'statistics': stats,
        'domain_info': {
            'domain_id': variant - 1,  # 0-indexed for CL frameworks
            'domain_name': f'phase_mask_variant_{variant}',
            'description': f'Shift-variant blur using phase mask variant {variant}',
        }
    }


# Single Image Mode

def process_single_image(args):
    psf_library = load_psf_library(args.library)
    image = load_image(args.input)

    sensor_fov = compute_sensor_fov_from_library(psf_library)
    print(f"Auto-computed sensor FOV: {sensor_fov[0]:.2f}° x {sensor_fov[1]:.2f}°")

    print(f"Patch size: {args.patch_size}px")
    print(f"Overlap ratio: {args.overlap_ratio:.0%}")

    result = overlap_add_convolution(
        image,
        psf_library,
        patch_size=args.patch_size,
        psf_crop_size=args.psf_crop_size,
        overlap_ratio=args.overlap_ratio,
        sensor_fov_degrees=sensor_fov,
        window_type=args.window_type,
        visualize_fields=args.visualize
    )

    if args.visualize:
        blurred_image, field_info = result
    else:
        blurred_image = result
        field_info = None

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_blurred{ext}"

    print(f"\nSaving blurred image: {args.output}")
    save_image(blurred_image, args.output)

    if args.visualize and field_info:        
        save_dir = args.save_viz if args.save_viz else os.path.dirname(args.output)
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(args.input))[0]

        field_save = os.path.join(save_dir, f"{base_name}_field_assignment.pdf")
        visualize_field_assignment(image, field_info, psf_library, save_path=field_save)

        compare_save = os.path.join(save_dir, f"{base_name}_comparison.pdf")
        visualize_comparison(image, blurred_image, save_path=compare_save)


# Dataset Generation Mode

def process_image_for_dataset(image_path, psf_library, output_blur_path, output_sharp_path, config):
    try:
        image = load_image(image_path)
        blurred = overlap_add_convolution(
            image,
            psf_library,
            patch_size=config['patch_size'],
            psf_crop_size=config['psf_crop_size'],
            overlap_ratio=config['overlap_ratio'],
            sensor_fov_degrees=config['sensor_fov'],
            window_type=config['window_type'],
            visualize_fields=False
        )

        save_image(blurred, output_blur_path)
        save_image(image, output_sharp_path)

        return True, os.path.basename(image_path)

    except Exception as e:
        return False, f"{os.path.basename(image_path)}: {str(e)}"


def generate_variant_dataset(variant, train_dir, val_dir, output_root, library_dir, config, num_workers=1):    
    print(f"\nGenerating Dataset for Variant {variant}")

    library_path = os.path.join(library_dir, f"psf_library_variant_{variant}.npz")
    psf_library = load_psf_library(library_path)

    config['sensor_fov'] = compute_sensor_fov_from_library(psf_library)
    print(f"Auto-computed sensor FOV: {config['sensor_fov'][0]:.2f}° x {config['sensor_fov'][1]:.2f}°")

    variant_dir = os.path.join(output_root, f"variant_{variant}")
    splits = {
        'train': {'input': train_dir, 'blur': None, 'sharp': None},
        'val': {'input': val_dir, 'blur': None, 'sharp': None}
    }

    for split_name in splits:
        splits[split_name]['blur'] = os.path.join(variant_dir, split_name, 'blur')
        splits[split_name]['sharp'] = os.path.join(variant_dir, split_name, 'sharp')
        os.makedirs(splits[split_name]['blur'], exist_ok=True)
        os.makedirs(splits[split_name]['sharp'], exist_ok=True)

    stats = {
        'train': {'total': 0, 'processed': 0, 'failed': 0, 'failed_files': []},
        'val': {'total': 0, 'processed': 0, 'failed': 0, 'failed_files': []}
    }

    for split_name, split_info in splits.items():
        input_dir = split_info['input']

        if not os.path.exists(input_dir):
            print(f"\nWarning: {split_name} directory not found: {input_dir}")
            continue

        print(f"\nProcessing {split_name} split...")
        image_paths = get_image_paths(input_dir)

        stats[split_name]['total'] = len(image_paths)
        print(f"Found {len(image_paths)} images")

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for img_path in image_paths:
                    filename = os.path.basename(img_path)
                    base = os.path.splitext(filename)[0]
                    out_filename = f"{base}.png"

                    blur_path = os.path.join(split_info['blur'], out_filename)
                    sharp_path = os.path.join(split_info['sharp'], out_filename)

                    future = executor.submit(
                        process_image_for_dataset,
                        img_path, psf_library, blur_path, sharp_path, config
                    )
                    futures[future] = img_path

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}"):
                    success, msg = future.result()
                    if success:
                        stats[split_name]['processed'] += 1
                    else:
                        stats[split_name]['failed'] += 1
                        stats[split_name]['failed_files'].append(msg)
        else:
            for img_path in tqdm(image_paths, desc=f"{split_name}"):
                filename = os.path.basename(img_path)
                base = os.path.splitext(filename)[0]
                out_filename = f"{base}.png"

                blur_path = os.path.join(split_info['blur'], out_filename)
                sharp_path = os.path.join(split_info['sharp'], out_filename)

                success, msg = process_image_for_dataset(
                    img_path, psf_library, blur_path, sharp_path, config
                )
                if success:
                    stats[split_name]['processed'] += 1
                else:
                    stats[split_name]['failed'] += 1
                    stats[split_name]['failed_files'].append(msg)

        print(f"Processed: {stats[split_name]['processed']}/{stats[split_name]['total']}")
        if stats[split_name]['failed'] > 0:
            print(f"Failed: {stats[split_name]['failed']}")

    metadata = create_metadata(variant, psf_library, config, stats)
    metadata_path = os.path.join(variant_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return stats


def generate_all_datasets(args):        
    if args.variant == 'all':
        variants = list(range(1, DEFAULT_CONFIG['num_variants'] + 1))
        print(f"\nGenerating Dataset for All {args.variant} Variants")
    else:
        variants = [int(args.variant)]

    print(f"\nVariants to process: {variants}")
    print(f"Train directory: {args.train_dir}")
    print(f"Val directory: {args.val_dir}")
    print(f"Output root: {args.output_root}")

    os.makedirs(args.output_root, exist_ok=True)

    config = {
        'patch_size': args.patch_size,
        'psf_crop_size': args.psf_crop_size,
        'overlap_ratio': args.overlap_ratio,
        'window_type': args.window_type,
    }

    all_stats = {}

    for variant in variants:
        stats = generate_variant_dataset(
            variant=variant,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_root=args.output_root,
            library_dir=args.library_dir,
            config=config.copy(),
            num_workers=args.num_workers
        )
        all_stats[f'variant_{variant}'] = stats

    dataset_info = {
        'created_at': datetime.now().isoformat(),
        'num_variants': len(variants),
        'variants': [f'variant_{v}' for v in variants],
        'source_datasets': {
            'train': args.train_dir,
            'val': args.val_dir,
        },
        'processing_config': config,
        'statistics': all_stats,
    }

    info_path = os.path.join(args.output_root, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Shift-Variant Blur Synthesis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # single image mode
    single_parser = subparsers.add_parser('single', help='Process a single image')
    single_parser.add_argument('--input', '-i', required=True, help='Input image path')
    single_parser.add_argument('--library', '-l', required=True, help='Path to PSF library .npz file')
    single_parser.add_argument('--output', '-o', default=None, help='Output path')
    single_parser.add_argument('--visualize', '-v', action='store_true',
                               help='Generate field assignment and comparison visualizations')
    single_parser.add_argument('--save-viz', default=None, help='Directory to save visualizations')
    single_parser.add_argument('--patch-size', type=int, default=DEFAULT_CONFIG['patch_size'],
                               help='Convolution patch size')
    single_parser.add_argument('--psf-crop-size', type=int, default=DEFAULT_CONFIG['psf_crop_size'],
                               help='Size to crop PSF before resizing')
    single_parser.add_argument('--overlap-ratio', type=float, default=DEFAULT_CONFIG['overlap_ratio'],
                               help='Overlap ratio between patches')
    single_parser.add_argument('--window-type', default=DEFAULT_CONFIG['window_type'], 
                               choices=['hann', 'tukey', 'bartlett'], help='Window function type')
    
    # dataset mode
    dataset_parser = subparsers.add_parser('dataset', help='Generate paired datasets for training')
    dataset_parser.add_argument('--train-dir', required=True, 
                                help='Directory containing training images (e.g., DIV2K_train_HR)')
    dataset_parser.add_argument('--val-dir', required=True,
                                help='Directory containing validation images (e.g., DIV2K_valid_HR)')
    dataset_parser.add_argument('--variant', default='all',
                                help='Variant number (1-6) or "all" for all variants')
    dataset_parser.add_argument('--output-root', default='datasets/shift_variant_blur',
                                help='Root output directory for generated datasets')
    dataset_parser.add_argument('--library-dir', default=DEFAULT_CONFIG['library_dir'],
                                help='Directory containing PSF library .npz files')
    dataset_parser.add_argument('--patch-size', type=int, default=DEFAULT_CONFIG['patch_size'],
                                help='Convolution patch size')
    dataset_parser.add_argument('--psf-crop-size', type=int, default=DEFAULT_CONFIG['psf_crop_size'],
                                help='Size to crop PSF before resizing')
    dataset_parser.add_argument('--overlap-ratio', type=float, default=DEFAULT_CONFIG['overlap_ratio'],
                                help='Overlap ratio between patches')
    dataset_parser.add_argument('--window-type', default=DEFAULT_CONFIG['window_type'],
                                choices=['hann', 'tukey', 'bartlett'], help='Window function type')
    dataset_parser.add_argument('--num-workers', type=int, default=1,
                                help='Number of parallel workers (1 for sequential)')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == 'single':
        process_single_image(args)
    elif args.mode == 'dataset':
        generate_all_datasets(args)


if __name__ == '__main__':
    main()
