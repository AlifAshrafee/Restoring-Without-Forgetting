import os
from psf_tools.library import load_psf_library
from psf_tools.psf_visualization import visualize_psf_grid


def main():
    target_dir = "star_psfs_separated"
    variants = [1, 2, 3, 4, 5, 6]
    save_dir = "outputs/psf_visualizations"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for variant in variants:
        psf_library = load_psf_library(os.path.join(target_dir, f"psf_library_variant_{variant}.npz"))
        save_path = os.path.join(save_dir, f"psf_grid_variant_{variant}.pdf")
        visualize_psf_grid(psf_library, crop_size=256, save_path=save_path)


if __name__ == '__main__':
    main()

