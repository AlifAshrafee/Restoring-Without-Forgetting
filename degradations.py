"""
Degradation synthesis functions for cross-task domain-incremental restoration.
All functions operate on float32 images in [0, 1] range, RGB channel order.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


### Gaussian Noise Synthesis

def add_gaussian_noise(img, sigma=None, seed=42):
    if sigma is None:
        sigma = np.random.uniform(15, 50)
    rng = np.random.RandomState(seed)
    noise = rng.randn(*img.shape).astype(np.float32) * (sigma / 255.0)
    return np.clip(img + noise, 0.0, 1.0)

### Motion Blur Synthesis

def _create_motion_kernel(length, angle_deg, curvature, kernel_size, num_pts=None):
    """
    Build a motion blur kernel by rasterizing a parametric trajectory of explicit
    length L (pixels) at angle theta, with optional sinusoidal perpendicular
    perturbation of amplitude C * L.

    Args:
        length: Trajectory length (motion magnitude) in pixels.
        angle_deg: Motion direction in degrees, [0, 360].
        curvature: Perpendicular curvature amplitude, 0 = linear, 0.6 = strongly curved.
        kernel_size: Side length of the kernel canvas (odd integer).
        num_pts: Number of trajectory sample points; defaults to ~10x length.
    """
    if num_pts is None:
        num_pts = max(int(length * 10), 200)

    s = np.linspace(-0.5, 0.5, num_pts)
    angle_rad = np.deg2rad(angle_deg)

    # base linear trajectory of length L along angle
    xs = length * s * np.cos(angle_rad)
    ys = length * s * np.sin(angle_rad)

    # perpendicular curvature for camera-shake-like paths
    if curvature > 0:
        perp_x, perp_y = -np.sin(angle_rad), np.cos(angle_rad)
        phase = np.random.uniform(0, 2 * np.pi)
        freq = np.random.uniform(0.5, 2.0)
        perturb = curvature * length * np.sin(2 * np.pi * freq * s + phase) * 0.25
        xs += perp_x * perturb
        ys += perp_y * perturb

    # center trajectory at origin
    xs = xs - xs.mean()
    ys = ys - ys.mean()

    # rasterize with sub-pixel bilinear deposition
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    for xi, yi in zip(xs, ys):
        xp, yp = xi + c, yi + c
        if 0 <= xp < kernel_size - 1 and 0 <= yp < kernel_size - 1:
            x0, y0 = int(np.floor(xp)), int(np.floor(yp))
            fx, fy = xp - x0, yp - y0
            kernel[y0, x0] += (1 - fx) * (1 - fy)
            kernel[y0, x0 + 1] += fx * (1 - fy)
            kernel[y0 + 1, x0] += (1 - fx) * fy
            kernel[y0 + 1, x0 + 1] += fx * fy

    total = kernel.sum()
    if total > 1e-8:
        kernel = kernel / total
    else:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[c, c] = 1.0
    return kernel


def add_motion_blur(img, length=None, angle_deg=None, curvature=None, add_noise=True):
    """
    Args:
        img: Clean image.
        add_noise: Adds mild post-blur Gaussian noise to mimic long-exposure handheld sensor noise.
    """
    if length is None:
        length = float(np.random.uniform(11, 41))
    if angle_deg is None:
        angle_deg = float(np.random.uniform(0, 360))
    if curvature is None:
        curvature = float(np.random.uniform(0.0, 0.6))

    # kernel size needs to contain the trajectory bounding box
    max_extent = length * np.sqrt(1.0 + (curvature * 0.25) ** 2)
    kernel_size = 2 * int(max_extent / 2 + 3) + 1

    kernel = _create_motion_kernel(length, angle_deg, curvature, kernel_size)

    # apply blur per channel via 2D convolution with reflection padding
    blurred = np.zeros_like(img)
    for c in range(3):
        blurred[:, :, c] = cv2.filter2D(
            img[:, :, c], -1, kernel, borderType=cv2.BORDER_REFLECT
        )

    # post-blur sensor noise (long-exposure / low-light handheld capture)
    if add_noise and np.random.rand() < 0.4:
        sigma = np.random.uniform(1.0, 4.0) / 255.0
        noise = np.random.randn(*blurred.shape).astype(np.float32) * sigma
        blurred = blurred + noise

    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


### Rain Streak Synthesis

def _create_rain_layer(h, w, num_streaks=800, streak_length_range=(20, 60),
                       streak_thickness_range=(1, 3), angle_range=(-30, 30)):
    """
    Generates a single rain streak layer as an alpha map.
    Streaks are near-vertical lines with directional motion-blur softening,
    following the Garg & Nayar (2007) / Yang et al. (2017) parametric model.
    """
    rain_layer = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_streaks):
        # random position
        x = np.random.randint(0, w)
        y = np.random.randint(-h // 4, h)

        # streak parameters
        length = np.random.randint(*streak_length_range)
        thickness = np.random.randint(*streak_thickness_range)
        angle = np.random.uniform(*angle_range)  # degrees from vertical

        # streak intensity (brighter streaks are rarer)
        intensity = np.random.uniform(0.4, 1.0)

        # computing endpoints
        angle_rad = np.deg2rad(angle)
        dx = int(length * np.sin(angle_rad))
        dy = int(length * np.cos(angle_rad))

        pt1 = (x, y)
        pt2 = (x + dx, y + dy)

        cv2.line(rain_layer, pt1, pt2, float(intensity), thickness, lineType=cv2.LINE_AA)

    # applying directional motion blur along the streak direction to soften
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[:, kernel_size // 2] = 1.0 / kernel_size
    rain_layer = cv2.filter2D(rain_layer, -1, kernel)

    return rain_layer


def add_rain(img, num_layers=None, streak_density=None, alpha=None, veil=None):
    """
    Synthesizes rain streaks via multi-layer parametric composition with
    atmospheric veiling, following the multi-layer formulation of Wei et al.
    (CVPR 2019). Each layer represents a depth plane; far layers are blurred and
    have shorter, thinner streaks (depth-of-field effect).
    Model:
        1. Streak composite: J_streak = J + alpha * SUM_l(S_l)
        2. Atmospheric veiling: I_rain = (1 - veil) * J_streak + veil * A_rain
    Args:
        img: Clean image.
        num_layers: Number of rain layers (depth planes).
        streak_density: Base number of streaks per layer.
        alpha: Global rain streak intensity.
        veil: Atmospheric veiling factor. Controls global
              contrast reduction from suspended droplet scattering.
    """
    h, w = img.shape[:2]

    if num_layers is None:
        num_layers = np.random.randint(3, 10)
    if streak_density is None:
        streak_density = np.random.randint(500, 2500)
    if alpha is None:
        alpha = np.random.uniform(0.5, 1.0)
    if veil is None:
        veil = np.random.uniform(0.0, 0.5)

    angle_bounds = np.sort(np.random.uniform(-30, 30, size=2))
    angle_range = (float(angle_bounds[0]), float(angle_bounds[1]))

    rain_accumulator = np.zeros((h, w), dtype=np.float32)

    for layer_idx in range(num_layers):
        # farther layers: thinner, shorter, more blurred streaks
        depth_factor = (layer_idx + 1) / num_layers

        layer_streaks = int(streak_density * (1.0 - 0.3 * depth_factor))
        length_range = (int(15 * (1.0 + 0.5 * depth_factor)), int(70 * (1.0 + 0.3 * depth_factor)))
        thickness_range = (1, max(2, int(3 * (1.0 - 0.3 * depth_factor))))

        layer = _create_rain_layer(h, w, layer_streaks, length_range, thickness_range, angle_range=angle_range)

        # depth-of-field blur for far layers
        if layer_idx > 0:
            blur_sigma = 0.5 + depth_factor * 1.5
            layer = gaussian_filter(layer, sigma=blur_sigma)

        rain_accumulator += layer * (1.0 - 0.2 * depth_factor)

    # streak composite
    rain_accumulator = np.clip(rain_accumulator, 0, 1)
    # slight blue tint to match real rain spectral response
    rain_rgb = rain_accumulator[:, :, None] * np.array([0.85, 0.85, 0.90], dtype=np.float32)
    streaked = np.clip(img + alpha * rain_rgb, 0.0, 1.0)

    # atmospheric veiling from suspended droplet scattering (global contrast reduction)
    A_rain = np.random.uniform(0.65, 0.85)  # gray atmospheric light from water droplet scattering
    rainy = (1.0 - veil) * streaked + veil * A_rain

    return np.clip(rainy, 0.0, 1.0).astype(np.float32)


### Haze Synthesis (Atmospheric Scattering Model)

_midas_cache = {}

def _get_midas_model():
    """
    Lazy-loads and caches the MiDaS DPT-Hybrid monocular depth estimation model.
    """
    if "model" not in _midas_cache:
        import torch
        import warnings

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[degradations] Loading MiDaS DPT-Hybrid on {device}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
            warnings.filterwarnings("ignore", category=UserWarning, module="timm")
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            model.to(device).eval()

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = transforms.dpt_transform

        _midas_cache["model"] = model
        _midas_cache["transform"] = transform
        _midas_cache["device"] = device
        print("[degradations] MiDaS loaded")

    return _midas_cache["model"], _midas_cache["transform"], _midas_cache["device"]


def _estimate_depth_midas(img):
    """
    Estimates a normalized depth [0, 1] map using MiDaS DPT-Hybrid monocular depth estimation model for scene-aware haze
    """
    import torch

    model, transform, device = _get_midas_model()
    h, w = img.shape[:2]

    img_uint8 = (img * 255).astype(np.uint8)
    input_batch = transform(img_uint8).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy().astype(np.float32)

    # MiDaS outputs inverse depth (higher = closer); invert for scattering
    depth = depth.max() - depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def add_haze(img, beta=None, A=None, depth_scale=None):
    """
    Synthesizes haze via the atmospheric scattering model (Koschmieder 1924):
        I(x) = J(x) * t(x) + A * (1 - t(x))
    where t(x) = exp(-beta * d(x)).

    Args:
        img: Clean image.
        beta: Scattering coefficient.
        A: Global atmospheric light.
        depth_scale: Multiplier mapping normalized MiDaS depth to physical-equiv meters.
    """
    try:
        depth_map = _estimate_depth_midas(img)
    except (ImportError, Exception) as e:
        if "model" not in _midas_cache:
            raise RuntimeError(f"[degradations] Pre-trained MiDaS model unavailable ({e})")

    if beta is None:
        beta = np.random.uniform(0.05, 0.8)

    # per-channel atmospheric light with slight chromatic variation
    if A is None:
        A_mean = np.random.uniform(0.75, 1.0)
        A = np.clip(A_mean + np.random.randn(3) * 0.05, 0.7, 1.0).astype(np.float32)
    elif np.isscalar(A):
        A = np.array([A, A, A], dtype=np.float32)
    else:
        A = np.asarray(A, dtype=np.float32)

    # wider depth scaling to mirror outdoor scene depth ranges
    if depth_scale is None:
        depth_scale = np.random.uniform(1.5, 3.5)

    d = depth_map * depth_scale + 0.5  # meters-equiv

    t = np.exp(-beta * d).astype(np.float32)
    t = np.clip(t, 0.05, 1.0)  # lower-bound to avoid division issues
    t = t[:, :, None]  # (H, W, 1) for broadcasting

    A_broadcast = A.reshape(1, 1, 3)
    hazy = img * t + A_broadcast * (1.0 - t)
    return np.clip(hazy, 0.0, 1.0).astype(np.float32)


### Low-Light Synthesis

def add_low_light(img, gamma=None, noise_scale=None):
    """
    Simulates low-light capture via gamma darkening followed by physically-grounded
    Poisson (shot) + Gaussian (read) sensor noise, following the SID (Chen et al.
    2018) sensor model.
    The degradation model follows the low-light imaging pipeline:
        1. Darken: I_dark = I_clean ^ gamma   (simulates reduced exposure)
        2. Shot noise: I_shot ~ Poisson(I_dark * k) / k
        3. Read noise: I_noisy = I_shot + N(0, sigma_read)
    Args:
        img: Clean image.
        gamma: Darkening gamma (> 1 means darker).
        noise_scale: Controls overall sensor noise magnitude.
    """
    if gamma is None:
        gamma = np.random.uniform(1.8, 4.5)
    if noise_scale is None:
        noise_scale = np.random.uniform(0.01, 0.06)

    # Gamma darkening
    darkened = np.power(np.clip(img, 1e-8, 1.0), gamma).astype(np.float32)

    # Poisson shot noise (signal-dependent)
    k = 1.0 / (noise_scale + 1e-8)  # scale factor to control the shot noise level
    noisy = np.random.poisson(np.clip(darkened * k, 0, None)).astype(np.float32) / k

    # Gaussian read noise (signal-independent)
    sigma_read = noise_scale * 0.5
    read_noise = np.random.randn(*img.shape).astype(np.float32) * sigma_read
    noisy = noisy + read_noise

    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


DEGRADATION_FUNCTIONS = {
    "deblur": add_motion_blur,
    "dehaze": add_haze,
    "derain": add_rain,
    "lowlight": add_low_light,
}

DOMAIN_ORDER = ["deblur", "dehaze", "derain", "lowlight"]
