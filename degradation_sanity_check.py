"""
Usage:
    python degradation_sanity_check.py --input_img ./path/to/image.png
"""

import os
import argparse
import numpy as np
import cv2

from degradations import add_motion_blur, add_rain, add_haze, add_low_light


def load_real_image(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def compute_psnr(gt, degraded):
    mse = np.mean((gt - degraded) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", type=str, default="./datasets/DIV2K/DIV2K_train_HR/0002.png")
    parser.add_argument("--output_dir", type=str, default="./outputs/degradations_sanity_check_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    gt = load_real_image(args.input_img)
    h, w = gt.shape[:2]
    img_bgr = cv2.cvtColor((np.clip(gt, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output_dir, "GT.png"), img_bgr)

    # defining degradations
    # # lowest params
    # degradations = [
    #     ("Deblur", "Motion Blur (L=11, θ=0°, C=0.0)", add_motion_blur, {"length": 11, "angle_deg": 0, "curvature": 0.0}),
    #     ("Derain", "Rain Streaks (L=3, n=500, α=0.5, veil=0.0)", add_rain, 
    #      {"num_layers": 3, "streak_density": 500, "alpha": 0.5, "veil": 0.0}),
    #     ("Dehaze", "Haze (β=0.05, A=0.75, D=1.5)", add_haze, {"beta": 0.05, "A": 0.75, "depth_scale": 1.5}),
    #     ("Lowlight", "Low-Light (γ=1.8, ns=0.01)", add_low_light, {"gamma": 1.8, "noise_scale": 0.01}),
    # ]

    # highest params
    degradations = [
        ("Deblur", "Motion Blur (L=41, θ=90°, C=0.6)", add_motion_blur, {"length": 41, "angle_deg": 90, "curvature": 0.6}),
        ("Derain", "Rain Streaks (L=10, n=2500, α=1.0, veil=0.5)", add_rain, 
         {"num_layers": 10, "streak_density": 2500, "alpha": 1.0, "veil": 0.5}),
        ("Dehaze", "Haze (β=0.8, A=1.0, D=3.5)", add_haze, {"beta": 0.8, "A": 1.0, "depth_scale": 3.5}),
        ("Lowlight", "Low-Light (γ=4.5, ns=0.06)", add_low_light, {"gamma": 4.5, "noise_scale": 0.06}),
    ]

    # applying each degradation
    results = []
    for domain_id, name, func, params in degradations:
        np.random.seed(args.seed)
        degraded = func(gt, **params)
        p = compute_psnr(gt, degraded)
        print(f"{domain_id}: {name} PSNR={p:.2f} dB")
        results.append((domain_id, name, p, params, degraded))

    # side-by-side comparison grid
    panel_h = 300
    ratio = panel_h / h
    panel_w = int(w * ratio)

    def make_panel(img_bgr, domain_id, psnr_val, params):
        resized = cv2.resize(img_bgr, (panel_w, panel_h))
        line1 = f"{domain_id} ({psnr_val:.1f}dB)"
        cv2.putText(resized, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        parts = [f"{param.split('_')[-1]}={val}" for param, val in params.items()]
        line2 = " ".join(parts)
        cv2.putText(resized, line2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1, cv2.LINE_AA)
        return resized

    panels = []

    for domain_id, _, psnr_val, params, degraded in results:
        img_bgr = cv2.cvtColor((np.clip(degraded, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        panels.append(make_panel(img_bgr, domain_id, psnr_val, params))

    top_row = np.concatenate(panels[:2], axis=1)
    bottom_row = np.concatenate(panels[2:], axis=1)
    grid = np.concatenate([top_row, bottom_row], axis=0)
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    cv2.imwrite(grid_path, grid)


if __name__ == "__main__":
    main()