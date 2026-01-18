import os
import cv2
from glob import glob

# Paths
clean_images_dir = 'dataset/clean_images_png/'       # clean Places2 .png images
masks_dir = 'dataset/masks/'                         # generated masks (black strokes on white)
output_images_dir = 'dataset/images/'                # masked images (input to model)
output_masks_dir = 'dataset/masks_processed/'        # binary mask (0=masked, 255=keep)

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

image_paths = sorted(glob(os.path.join(clean_images_dir, '*.png')))

for image_path in image_paths:
    base = os.path.basename(image_path).split('.')[0]
    mask_path = os.path.join(masks_dir, f"{base}_mask.png")

    if not os.path.exists(mask_path):
        print(f"Skipping {base} (no mask found)")
        continue

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert white=255 to 1, black=0 stays 0, and invert to create binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply mask: make masked (black stroke) regions 0 in input image
    masked_image = image.copy()
    masked_image[binary_mask == 0] = 0

    # Save masked input and binary mask
    cv2.imwrite(os.path.join(output_images_dir, f"{base}.png"), masked_image)
    cv2.imwrite(os.path.join(output_masks_dir, f"{base}.png"), 255 - binary_mask)  # invert for training

    print(f"Processed: {base}")
