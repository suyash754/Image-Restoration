import argparse
from distutils.util import strtobool
import os

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.model import PConvUNet


def main(args):
    # Define the used device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(finetune=False, layer_size=7)
    model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    model.to(device)
    model.eval()

    # Derive mask name from input image name
    img_name = os.path.basename(args.img)
    mask_name = os.path.splitext(img_name)[0] + "_mask.png"
    mask_path = os.path.join(args.mask, mask_name)

    # Loading Input and Mask
    print("Loading the inputs...")
    org = Image.open(args.img)
    org = TF.to_tensor(org.convert('RGB'))

    mask = Image.open(args.mask)  # directly use provided path
    mask = TF.to_tensor(mask.convert('RGB'))



    inp = org * mask

    # Model prediction
    print("Model Prediction...")
    with torch.no_grad():
        inp_ = inp.unsqueeze(0).to(device)
        mask_ = mask.unsqueeze(0).to(device)
        if args.resize:
            org_size = inp_.shape[-2:]
            inp_ = F.interpolate(inp_, size=256)
            mask_ = F.interpolate(mask_, size=256)
        raw_out, _ = model(inp_, mask_)

    if args.resize:
        raw_out = F.interpolate(raw_out, size=org_size)

    # Post process
    raw_out = raw_out.to(torch.device('cpu')).squeeze()
    raw_out = raw_out.clamp(0.0, 1.0)
    out = mask * inp + (1 - mask) * raw_out

    # Saving the output
    print("Saving the output...")
    out = TF.to_pil_image(out)
    output_path = os.path.join("output", f"restored_{img_name}")
    os.makedirs("output", exist_ok=True)
    out.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--img', type=str, default="input_images/jkl.png", help="Path to input (masked) image")
    parser.add_argument('--mask', type=str, required=True, help="Path to the mask image")
    parser.add_argument('--model', type=str, default="ckpt/latest.pth", help="Trained model checkpoint")
    parser.add_argument('--checkpoint', type=str, default="cktp/models/pretrained_pconv.pth")
    parser.add_argument('--resize', type=strtobool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    main(args)