import numpy as np
from scipy import signal
from scipy.ndimage import zoom


def create_window_2d(size, window_type='hann', tukey_alpha=0.5):
    if window_type == 'hann':
        window_1d = signal.windows.hann(size, sym=True)
    elif window_type == 'tukey':
        window_1d = signal.windows.tukey(size, alpha=tukey_alpha, sym=True)
    elif window_type == 'bartlett':
        window_1d = signal.windows.bartlett(size)
    else:
        raise ValueError(f"Unknown window type: {window_type}. "
                        f"Supported: 'hann', 'tukey', 'bartlett'")

    window_2d = np.outer(window_1d, window_1d)

    return window_2d


def resize_psf(psf_rgb, target_size):
    H, W, _ = psf_rgb.shape
    if H == target_size and W == target_size:
        return psf_rgb

    zoom_factor = target_size / H
    resized = np.zeros((target_size, target_size, 3), dtype=np.float32)
    for c in range(3):
        resized[:, :, c] = zoom(psf_rgb[:, :, c], zoom_factor, order=1)
        resized[:, :, c] = resized[:, :, c] / (np.sum(resized[:, :, c]) + 1e-10)

    return resized


def crop_psf_center(rgb_psf, crop_size):
    H, W = rgb_psf.shape[:2]
    center_y, center_x = H // 2, W // 2
    half_crop = crop_size // 2

    y1 = max(0, center_y - half_crop)
    y2 = min(H, center_y + half_crop)
    x1 = max(0, center_x - half_crop)
    x2 = min(W, center_x + half_crop)

    rgb_psf_crop = rgb_psf[y1:y2, x1:x2, :]

    return rgb_psf_crop

