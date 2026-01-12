import os
import sys
import argparse
from dataset import default_loader
import imageio
import numpy as np
from skimage.transform import resize
import torch
import torchvision.transforms as transforms
from PIL import Image

from model.encoder_gcn import MEFARG
from utils import image_test, load_state_dict
from conf import get_config

def load_model(conf, model_path):
    conf.agg = "original"
    net = MEFARG(num_classes=15, backbone="swin_transformer_base", numEncoderLayers=3, numLandmarks=66, conf=conf)
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No checkpoint found at '{model_path}'")
        
    net = load_state_dict(net, model_path)
    net.eval()
    return net

def load_image(image_path):
    # 6. Load and preprocess image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        raise FileNotFoundError(f"Image file not found at {image_path}")

    loader = default_loader
    transform = image_test(crop_size=224)

    image = loader(image_path)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor        

def main():
    # 1. Parse arguments specifically for this script
    parser = argparse.ArgumentParser(description='Run AU detection on a single image')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    
    # 2. Separate run.py specific args from potential config args
    args, remaining_args = parser.parse_known_args()
    
    # 3. Patch sys.argv so get_config() receives the remaining configuration arguments
    sys.argv = [sys.argv[0]] + remaining_args
    conf = get_config()
    
    # 4. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 5. Load model
    try:
        net = load_model(conf, args.checkpoint)
        net = net.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Preprocess Image
    input_tensor = load_image(args.img)
    input_tensor = input_tensor.to(device)

    # 7. Run Inference
    print(f"Running inference on {args.img}...")
    with torch.no_grad():
        outputs, _, _, _ = net(input_tensor)
    
    print("\n" + "="*45)
    print(f" Predictions")
    print("="*45)
    print(f"{'AU Index':<10} | {'Probability':<12} | {'State':<10}")
    print("-" * 45)
    
    for i, p in enumerate(outputs[0].cpu().numpy()):
        state = "Active" if p > 0.5 else "Inactive"
        print(f"{i:<10} | {p:.4f}       | {state:<10}")
    print("="*45 + "\n")

if __name__ == '__main__':
    main()
