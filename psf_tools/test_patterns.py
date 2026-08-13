import numpy as np


def generate_point_source_grid(image_size=(2048, 2048), grid_spacing=256, point_width=3, intensity=1.0):
    H, W = image_size
    image = np.zeros((H, W, 3), dtype=np.float32)

    y_positions = np.arange(grid_spacing//2, H, grid_spacing)
    x_positions = np.arange(grid_spacing//2, W, grid_spacing)

    half_width = point_width // 2

    for y in y_positions:
        for x in x_positions:
            y1 = max(0, y - half_width)
            y2 = min(H, y + half_width + 1)
            x1 = max(0, x - half_width)
            x2 = min(W, x + half_width + 1)

            image[y1:y2, x1:x2, :] = intensity

    print(f"Generated {len(y_positions)}x{len(x_positions)} = {len(y_positions) * len(x_positions)} point sources")

    return image