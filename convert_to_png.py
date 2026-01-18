import os
import cv2
from glob import glob

input_dir = 'dataset/clean_images/'
output_dir = 'dataset/clean_images_png/'

os.makedirs(output_dir, exist_ok=True)

jpg_images = glob(os.path.join(input_dir, '*.jpg'))

for img_path in jpg_images:
    img = cv2.imread(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.png")
    cv2.imwrite(output_path, img)
    print(f"Converted {filename}.jpg to {filename}.png")
