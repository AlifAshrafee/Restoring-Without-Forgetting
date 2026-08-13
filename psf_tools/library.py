import os
import numpy as np
from .parser import parse_psf_locations, read_rgb_psf_kernels


def build_psf_library(target_dir, variation_number, dead_fields=None, output_path=None):
    psf_locations = parse_psf_locations(target_dir, variation_number, dead_fields=dead_fields)
    num_fields = len(psf_locations)

    if num_fields == 0:
        raise ValueError(f"No valid fields found for variant {variation_number}")

    field_numbers = sorted(psf_locations.keys())
    field_positions = np.array([psf_locations[f] for f in field_numbers], dtype=np.float32)

    psf_kernels = {}
    for field in field_numbers:
        rgb_psf = read_rgb_psf_kernels(target_dir, variation_number, field)
        psf_kernels[field] = rgb_psf

    save_dict = {
        'variation_number': variation_number,
        'field_numbers': np.array(field_numbers, dtype=np.int32),
        'field_positions_degrees': field_positions,
    }

    for field in field_numbers:
        save_dict[f'psf_field_{field}'] = psf_kernels[field]

    print(f"Library summary:")
    print(f"Variant: {variation_number}")
    print(f"Number of Fields: {num_fields}")
    print(f"Each PSF kernel shape: {psf_kernels[field_numbers[0]].shape}")
    print(f"Field extent:")
    print(f"X=[{field_positions[:, 0].min():.3f}, {field_positions[:, 0].max():.3f}] degrees")
    print(f"Y=[{field_positions[:, 1].min():.3f}, {field_positions[:, 1].max():.3f}] degrees")

    if output_path is None:
        output_path = os.path.join(target_dir, f"psf_library_variant_{variation_number}.npz")

    np.savez_compressed(output_path, **save_dict)
    print(f"PSF Library Saved to: {output_path}")


def load_psf_library(library_path):
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"PSF library not found: {library_path}")

    data = np.load(library_path)
    variation_number = int(data['variation_number'])
    field_numbers = data['field_numbers']
    field_positions = data['field_positions_degrees']
    psf_kernels = {}

    for field in field_numbers:
        psf_kernels[field] = data[f'psf_field_{field}']

    psf_library = {
        'variation_number': variation_number,
        'field_numbers': field_numbers,
        'field_positions_degrees': field_positions,
        'psf_kernels': psf_kernels
    }

    print(f"Loaded PSF library for variant: {psf_library['variation_number']}")

    return psf_library

