import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def visualize_field_assignment(image, field_info, psf_library, save_path=None):
    if image.max() > 1.0:
        image = image / 255.0

    _, axes = plt.subplots(1, 3, figsize=(20, 7))

    variant = psf_library['variation_number']
    field_numbers = psf_library['field_numbers']
    field_positions = psf_library['field_positions_degrees']

    x_min, x_max = field_positions[:, 0].min(), field_positions[:, 0].max()
    y_min, y_max = field_positions[:, 1].min(), field_positions[:, 1].max()

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

    # image plot with field overlay
    axes[1].imshow(image)
    for info in field_info:
        cx, cy = info['center']
        field = info['field']
        axes[1].plot(cx, cy, 'ro')
        axes[1].text(cx, cy, f'F{field}', fontsize=7, color='yellow', ha='center', va='bottom',
                    bbox=dict(boxstyle='round, pad=0.2', facecolor='black'))

    axes[1].set_title('Field Numbers Overlay', fontsize=14)
    axes[1].axis('off')

    # field usage histogram
    field_counts = Counter([info['field'] for info in field_info])
    fields = sorted(field_counts.keys())
    counts = [field_counts[f] for f in fields]
    axes[2].bar(fields, counts, color='skyblue', edgecolor='navy', alpha=0.8)
    axes[2].set_xlabel('Field Number', fontsize=12)
    axes[2].set_ylabel('# times Applied', fontsize=12)
    axes[2].set_title('PSF Field Usage Frequency', fontsize=14)

    plt.suptitle('PSF Field Assignment Visualization', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def visualize_comparison(original, blurred, crop_coords=None, save_path=None):
    if original.max() > 1.0:
        original = original / 255.0
    if blurred.max() > 1.0:
        blurred = blurred / 255.0

    H, W = original.shape[:2]

    if crop_coords is None:
        crop_size = min(H, W) // 4
        y = H // 2 - crop_size // 2
        x = W // 2 - crop_size // 2
    else:
        y, x, crop_size = crop_coords

    orig_crop = original[y:y+crop_size, x:x+crop_size]
    blur_crop = blurred[y:y+crop_size, x:x+crop_size]

    _, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    # full images
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14)
    rect = plt.Rectangle((x, y), crop_size, crop_size, linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].axis('off')

    axes[1].imshow(blurred)
    axes[1].set_title('Shift-Variant Blurred', fontsize=14)
    rect = plt.Rectangle((x, y), crop_size, crop_size, linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].axis('off')

    # detail crops
    axes[2].imshow(orig_crop)
    axes[2].set_title('Original Image (Detail)', fontsize=14)
    axes[2].axis('off')

    axes[3].imshow(blur_crop)
    axes[3].set_title('Shift-Variant Blurred (Detail)', fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def visualize_point_source_convolution(original_point_grid, blurred_point_grid, save_path=None):
    _, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(original_point_grid)
    axes[0].set_title('Original Point Sources')
    axes[0].axis('off')

    blurred_point_grid = blurred_point_grid / (blurred_point_grid.max() + 1e-10)
    axes[1].imshow(np.sqrt(blurred_point_grid))
    axes[1].set_title('After Shift-Variant PSF (Normalized for visualization)')
    axes[1].axis('off')

    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')