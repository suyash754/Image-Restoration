import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# Paths
source_dir = 'clean_images/'        # Folder of clean 256x256 images
output_img_dir = 'dataset/input/'
output_mask_dir = 'dataset/mask/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

def generate_noise_mask(image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for _ in range(np.random.randint(5, 15)):
        x1, y1 = np.random.randint(0, 200, 2)
        x2, y2 = np.random.randint(56, 256, 2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def apply_mask_noise(image, mask):
    noisy = image.copy()
    noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    mask_3ch = cv2.merge([mask]*3)
    noisy = np.where(mask_3ch == 255, noise, noisy)
    return noisy

images = glob(os.path.join(source_dir, "*.jpg"))

for img_path in tqdm(images):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    
    mask = generate_noise_mask(img.shape)
    noisy_img = apply_mask_noise(img, mask)

    base = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_img_dir, base), noisy_img)
    cv2.imwrite(os.path.join(output_mask_dir, base), mask)
