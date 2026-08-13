import numpy as np
import matplotlib.pyplot as plt


def visualize_psf_grid(psf_library, crop_size=128, save_path=None):
    variant = psf_library['variation_number']
    field_numbers = psf_library['field_numbers']
    field_positions = psf_library['field_positions_degrees']

    x_positions = field_positions[:, 0]
    y_positions = field_positions[:, 1]

    unique_x = np.unique(x_positions)
    unique_y = np.unique(y_positions)

    x_min, x_max = x_positions.min(), x_positions.max()
    y_min, y_max = y_positions.min(), y_positions.max()

    # mapping positions to grid indices
    x_to_idx = {x: i for i, x in enumerate(sorted(unique_x))}
    y_to_idx = {y: i for i, y in enumerate(sorted(unique_y, reverse=True))}  # reversed for image coordinates

    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    # field layout plot
    square = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, linestyle='--')
    axes[0].add_patch(square)

    axes[0].scatter(
        field_positions[:, 0], field_positions[:, 1], 
        c=['red' if i == 0 and j == 0 else 'green' for i, j in zip(field_positions[:, 0], field_positions[:, 1])], 
        alpha=0.7, linewidth=1
        )

    for i, field_num in enumerate(field_numbers):
        axes[0].annotate(f'{field_num}', (field_positions[i, 0] + 0.1, field_positions[i, 1] + 0.1), fontsize=8)

    axes[0].grid(True, alpha=0.3, linestyle='--')    
    axes[0].set_xlabel('Field X (degrees)', fontsize=12)
    axes[0].set_ylabel('Field Y (degrees)', fontsize=12)
    axes[0].set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1))
    axes[0].set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 1))
    axes[0].set_title(f'PSF Field Layout: Variant - {variant}, Fields - {len(field_numbers)}', fontsize=14)

    # psf mosaic plot
    grid_height = len(unique_y)
    grid_width = len(unique_x)

    # creating empty canvas
    canvas_height = grid_height * crop_size
    canvas_width = grid_width * crop_size
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)

    # placing each PSF in the canvas
    for idx, field in enumerate(field_numbers):
        rgb_psf = psf_library['psf_kernels'][field]
        pos = field_positions[idx]

        # getting grid position
        grid_x = x_to_idx[pos[0]]
        grid_y = y_to_idx[pos[1]]

        # cropping to center of PSF
        H, W = rgb_psf.shape[:2]
        center_y, center_x = H // 2, W // 2
        half_crop = crop_size // 2

        y1 = max(0, center_y - half_crop)
        y2 = min(H, center_y + half_crop)
        x1 = max(0, center_x - half_crop)
        x2 = min(W, center_x + half_crop)

        rgb_crop = rgb_psf[y1:y2, x1:x2, :]

        # normalizing for displaying
        rgb_crop = rgb_crop / (rgb_crop.max() + 1e-10)

        # placing in canvas
        canvas_y1 = grid_y * crop_size
        canvas_y2 = canvas_y1 + rgb_crop.shape[0]
        canvas_x1 = grid_x * crop_size
        canvas_x2 = canvas_x1 + rgb_crop.shape[1]

        canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2, :] = rgb_crop

    # applying sqrt for better visualization
    canvas_display = np.sqrt(canvas)
    # canvas_display = np.power(canvas_display, 0.5)
    # canvas_display = np.clip(canvas_display, 0, 1.0)

    axes[1].imshow(canvas_display)
    axes[1].set_title(f'Spatially Arranged PSF Sensor Map Mosaic: {grid_height} x {grid_width} grid, '
                 f'Variant: {variant}, Fields: {len(field_numbers)}')

    # adding grid lines
    for i in range(1, grid_width):
        axes[1].axvline(x=i*crop_size, color='white', linewidth=0.5, alpha=0.3)
    for i in range(1, grid_height):
        axes[1].axhline(y=i*crop_size, color='white', linewidth=0.5, alpha=0.3)

    # adding position labels
    for idx, field in enumerate(field_numbers):
        pos = field_positions[idx]
        grid_x = x_to_idx[pos[0]]
        grid_y = y_to_idx[pos[1]]

        text_x = grid_x * crop_size + crop_size // 2
        text_y = grid_y * crop_size + crop_size - 10

        axes[1].text(text_x, text_y, f'F{field}',
               fontsize=8, color='yellow', ha='center', va='bottom',
               bbox=dict(boxstyle='round, pad=0.3', facecolor='black', alpha=0.5))

    axes[1].axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

