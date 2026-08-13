import os
from psf_tools.library import load_psf_library
from psf_tools.test_patterns import generate_point_source_grid
from psf_tools.overlap_add_conv import overlap_add_convolution, compute_sensor_fov_from_library
from psf_tools.convolution_visualization import visualize_point_source_convolution


def main():
    target_dir = "star_psfs_separated"
    variants = [1, 2, 3, 4, 5, 6]
    save_dir = "outputs/point_source_visualizations"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    point_grid = generate_point_source_grid(
        image_size=(2040, 2040),
        grid_spacing=256,
        point_width=10,
        intensity=1.0
    )

    for variant in variants:
        psf_library = load_psf_library(os.path.join(target_dir, f"psf_library_variant_{variant}.npz"))
        sensor_fov = compute_sensor_fov_from_library(psf_library)
        print(f"Auto-computed sensor FOV: {sensor_fov[0]:.2f}° x {sensor_fov[1]:.2f}°")
        save_path = os.path.join(save_dir, f"point_source_grid_variant_{variant}.pdf")

        blurred_grid = overlap_add_convolution(
            point_grid,
            psf_library,
            patch_size=512,
            psf_crop_size=512,
            overlap_ratio=0.5,
            sensor_fov_degrees=sensor_fov
            )

        visualize_point_source_convolution(point_grid, blurred_grid, save_path=save_path)
        print()


if __name__ == '__main__':
    main()

