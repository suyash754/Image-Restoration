from glob import glob
from PIL import Image
import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transform


class Places2(Dataset):
    def __init__(self, input_dir, mask_dir, img_transform, mask_transform):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.input_dir = input_dir
        self.mask_dir = mask_dir

        self.image_paths = glob(os.path.join(self.input_dir, '*.jpg'))
        self.image_paths.extend(glob(os.path.join(self.input_dir, '*.png')))
        self.image_paths.sort()

        self.mask_paths = []
        for img_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, f'{base_name}_mask.png')
            self.mask_paths.append(mask_path)

        self.N_mask = len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        img = self._load_img(img_path)
        img = self.img_transform(img.convert('RGB'))
        mask = Image.open(mask_path).convert('RGB')
        mask = self.mask_transform(mask)

        return img * mask, mask, img

    def _load_img(self, path):
        try:
            img = Image.open(path)
        except:
            extension = path.split('.')[-1]
            for i in range(10):
                new_path = path.split('.')[0][:-1] + str(i) + '.' + extension
                try:
                    img = Image.open(new_path)
                    break
                except:
                    continue
        return img

if __name__ == '__main__':
    # Example usage: Replace with your actual directory paths and transforms
    data_root = r'C:\Users\suyas\OneDrive\Desktop\partialconv-master\partialconv-master'  # Replace with the root directory containing input_images and masked_images
    input_image_folder = "C:/Users/suyas/OneDrive/Desktop/partialconv-master/partialconv-master/dataset/images"
    masked_image_folder = "C:/Users/suyas/OneDrive/Desktop/partialconv-master/partialconv-master/dataset/masks"

    # Example transforms (replace with your actual transforms)
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create the dataset instance
    dataset = Places2(input_image_folder, masked_image_folder, img_transform, mask_transform)

    # Example of accessing an item
    if len(dataset) > 0:
        masked_img, mask, original_img = dataset[0]
        print("Masked Image shape:", masked_img.shape)
        print("Mask shape:", mask.shape)
        print("Original Image shape:", original_img.shape)



