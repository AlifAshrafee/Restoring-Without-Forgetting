import os
import re
import glob
import numpy as np


def parse_psf_locations(target_dir, variation_number, dead_fields=None):
    psf_locations = {}
    psfs_dir = os.path.join(target_dir, f"variant_{variation_number}")

    if not os.path.exists(psfs_dir):
        raise FileNotFoundError(f"Variant directory not found: {psfs_dir}")

    for file_name in glob.glob(os.path.join(psfs_dir, "*.txt")):
        # expected format: PSF_Field{N}_Wavelength{M}_filenumber{V}_.txt
        field_number = int(file_name.split("Field")[-1].split("_")[0])
        if field_number in psf_locations:
            continue
        if dead_fields and field_number in dead_fields:
            continue

        with open(file_name, 'r', encoding='utf-16') as f:
            content = f.read()
        lines = content.strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            # extracting field position. format: "z.zzzz µm at x.xxxx, y.yyyy"
            if 'µm at' in line and 'deg' in line:
                match = re.search(r'([\d.]+)\s*µm at\s*([-\d.]+),\s*([-\d.]+)\s*', line)
                if match:
                    field_x_deg, field_y_deg = float(match.group(2)), float(match.group(3))
                    if field_number not in psf_locations:
                        psf_locations[field_number] = (field_x_deg, field_y_deg)

    return psf_locations


def parse_psf_kernel(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PSF file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()
    lines = content.strip().split('\n')

    data_start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line and re.match(r'^\s*[-+]?\d+\.?\d*[Ee][-+]?\d+', line):
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError(f"Could not find numeric data in {file_path}")

    psf = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue

        if line:
            row_values = []
            for val in line.split('\t'):
                val = val.strip()
                if val:
                    try:
                        row_values.append(float(val))
                    except ValueError:
                        continue
            if row_values:
                psf.append(row_values)

    if not psf:
        raise ValueError(f"No valid PSF data parsed from {file_path}")

    return np.array(psf, dtype=np.float32)


def read_rgb_psf_kernels(target_dir, variation_number, field):
    variant_dir = os.path.join(target_dir, f"variant_{variation_number}")
    wavelength_files = {}

    for wl in [1, 2, 3]:
        file_path = os.path.join(
            variant_dir,
            f"PSF_Field{field}_Wavelength{wl}_filenumber{variation_number}_.txt"
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PSF file for wavelength {wl} not found: {file_path}")
        wavelength_files[wl] = file_path

    psf_kernel_b = parse_psf_kernel(wavelength_files[1])
    psf_kernel_g = parse_psf_kernel(wavelength_files[2])
    psf_kernel_r = parse_psf_kernel(wavelength_files[3])

    psf_kernel_r = psf_kernel_r / (np.sum(psf_kernel_r) + 1e-10)
    psf_kernel_g = psf_kernel_g / (np.sum(psf_kernel_g) + 1e-10)
    psf_kernel_b = psf_kernel_b / (np.sum(psf_kernel_b) + 1e-10)

    rgb_psf_kernel = np.stack([psf_kernel_r, psf_kernel_g, psf_kernel_b], axis=-1)

    return rgb_psf_kernel

