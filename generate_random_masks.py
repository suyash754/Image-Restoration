import os
import cv2
import numpy as np
from glob import glob

# Settings
input_dir = 'dataset/clean_images_png/'
mask_output_dir = 'dataset/masks/'
image_size = 256
num_strokes = 5

os.makedirs(mask_output_dir, exist_ok=True)

def random_brush_mask(height, width):
    mask = np.ones((height, width), np.uint8) * 255  # white background
    for _ in range(num_strokes):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        thickness = np.random.randint(10, 30)
        cv2.line(mask, (x1, y1), (x2, y2), 0, thickness)
    return mask

image_paths = sorted(glob(os.path.join(input_dir, '*.png')))

for image_path in image_paths:
    filename = os.path.basename(image_path).split('.')[0]
    mask = random_brush_mask(image_size, image_size)
    mask_path = os.path.join(mask_output_dir, f"{filename}_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"Generated mask for {filename}")
