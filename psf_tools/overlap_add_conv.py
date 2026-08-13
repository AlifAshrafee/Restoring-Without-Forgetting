import numpy as np
from scipy import signal
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from .utils import create_window_2d, resize_psf, crop_psf_center


def overlap_add_convolution(image, psf_library, patch_size=256, psf_crop_size=None, overlap_ratio=0.5, 
                            sensor_fov_degrees=(36.0, 24.0), window_type='hann', visualize_fields=False):
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    H_orig, W_orig, C = image.shape

    pad_size = patch_size // 2
    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    H, W, C = image.shape

    print(f"Original image size: {H_orig}x{W_orig}")
    print(f"Padded image size: {H}x{W} ({pad_size}px per side)")

    step_size = int(patch_size * (1 - overlap_ratio))
    num_patches_y = int(np.ceil((H - patch_size) / step_size)) + 1
    num_patches_x = int(np.ceil((W - patch_size) / step_size)) + 1

    output = np.zeros_like(image, dtype=np.float64)
    weight_map = np.zeros((H, W, C), dtype=np.float64)

    window_2d = create_window_2d(patch_size, window_type=window_type)
    window_2d = np.repeat(window_2d[:, :, np.newaxis], C, axis=2)

    kdtree = cKDTree(psf_library['field_positions_degrees'])

    fov_x, fov_y = sensor_fov_degrees
    pixel_to_deg_x = fov_x / W
    pixel_to_deg_y = fov_y / H
    center_x, center_y = W / 2, H / 2

    print(f"Processing: {patch_size}px patches, {overlap_ratio:.0%} overlap")
    print(f"Patch grid: {num_patches_y} x {num_patches_x} = {num_patches_y * num_patches_x} patches")
    print(f"Sensor FOV: {fov_x:.1f}° x {fov_y:.1f}°")

    field_info = [] if visualize_fields else None  # stores (patch_center, field_number, sensor_pos)
    patch_count = 0

    for i in range(num_patches_y):
        y_start = min(i * step_size, H - patch_size)
        y_end = y_start + patch_size

        for j in range(num_patches_x):
            x_start = min(j * step_size, W - patch_size)
            x_end = x_start + patch_size

            actual_patch = image[y_start:y_end, x_start:x_end, :]

            patch_center_x = x_start + patch_size / 2
            patch_center_y = y_start + patch_size / 2

            pos_deg_x = (patch_center_x - center_x) * pixel_to_deg_x
            pos_deg_y = (center_y - patch_center_y) * pixel_to_deg_y  # flip y

            _, nearest_idx = kdtree.query((pos_deg_x, pos_deg_y), k=1)
            field_number = psf_library['field_numbers'][nearest_idx]
            psf = psf_library['psf_kernels'][field_number]

            if psf_crop_size is not None:
                psf = crop_psf_center(psf, psf_crop_size)
            psf = resize_psf(psf, patch_size)

            if visualize_fields:
                orig_center_x = patch_center_x - pad_size
                orig_center_y = patch_center_y - pad_size
                field_info.append({
                    'center': (orig_center_x, orig_center_y),
                    'field': field_number,
                    'sensor_pos': (pos_deg_x, pos_deg_y)
                })

            blurred_patch = np.zeros_like(actual_patch)
            for c in range(C):
                blurred_patch[:, :, c] = signal.fftconvolve(
                    actual_patch[:, :, c], 
                    psf[:, :, c], 
                    mode='same'
                )

            windowed_patch = blurred_patch * window_2d
            output[y_start:y_end, x_start:x_end, :] += windowed_patch
            weight_map[y_start:y_end, x_start:x_end, :] += window_2d

            patch_count += 1

    output = output / (weight_map + 1e-10)
    output = output[pad_size:pad_size+H_orig, pad_size:pad_size+W_orig, :]
    output = np.clip(output, 0, 1.0).astype(np.float32)

    print(f"Total patches processed: {patch_count}")
    print(f"Output shape: {output.shape[0]}x{output.shape[1]}")

    if visualize_fields:
        return output, field_info
    return output


def compute_sensor_fov_from_library(psf_library):
    """
    Computes sensor FOV bounds automatically determine sensor_fov_degrees parameter
    based on the extent of the PSF field positions.
    """
    positions = psf_library['field_positions_degrees']
    x_extent = positions[:, 0].max() - positions[:, 0].min()
    y_extent = positions[:, 1].max() - positions[:, 1].min()

    # small margin to ensure edge fields are reachable
    return (x_extent * 1.1, y_extent * 1.1)

