#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# from model import Autoencoder  # Ensure your model.py defines Autoencoder
from model import Autoencoder

def denormalize(t):
    """Denormalizes tensor from [-1, 1] to [0, 1]."""
    return t * 0.5 + 0.5

def process_image(model, device, image_path, transform):
    """
    Processes a single image through the model.
    Returns:
      - original_full: the original full-resolution grayscale image (as a NumPy array)
      - reconstructed_img: the model's reconstructed image at the resized resolution (for reference)
      - diff_upscaled: the absolute difference between original and reconstruction,
                       upscaled to the original resolution.
    """
    # Load the full-resolution image in grayscale
    pil_image = Image.open(image_path).convert("L")
    original_full = np.array(pil_image)  # Preserve the original size
    
    # Resize image for model input (e.g., 256x2048) using your transform
    input_tensor = transform(pil_image).unsqueeze(0).to(device)  # shape: (1, 1, 256, 2048)
    
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
    
    # Get the resized images (model input resolution)
    original_resized = denormalize(input_tensor.cpu()[0, 0]).numpy()  # shape: (256, 2048)
    reconstructed_img = denormalize(reconstructed_tensor.cpu()[0, 0]).numpy()
    
    # Compute the absolute difference at the resized resolution
    diff = np.abs((original_resized - reconstructed_img)*2)
    
    # Upscale the diff image to the original image size
    orig_h, orig_w = original_full.shape
    diff_upscaled = cv2.resize(diff, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    reconstructed_img_upscaled = cv2.resize(reconstructed_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return original_full, reconstructed_img_upscaled, diff_upscaled

def create_anomaly_overlay(original_img, diff, alpha=0.5):
    """
    Creates an anomaly overlay by thresholding the difference image.
    Regions with a difference above (mean + std) are colored red.
    """
    threshold = diff.mean() + diff.std()
    anomaly_mask = diff > threshold

    # Convert grayscale image to RGB for visualization.
    original_rgb = np.stack([original_img] * 3, axis=-1)
    overlay = original_rgb.copy()
    overlay[anomaly_mask, 0] = 1.0  # Red channel
    overlay[anomaly_mask, 1] = 0.0  # Remove green
    overlay[anomaly_mask, 2] = 0.0  # Remove blue
    blended = (1 - alpha) * original_rgb + alpha * overlay
    return blended, anomaly_mask

def save_compared(original_img, reconstructed_img, diff, img_path, args):
    """
    Create a comparison plot of the original image and the diff image,
    and save it as a PNG file.
    """
    fig, axs = plt.subplots(3, figsize=(20, 10))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    im = axs[1].imshow(reconstructed_img, cmap='gray')
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")
    
    im = axs[2].imshow(diff, cmap='hot')
    axs[2].set_title("Absolute Difference")
    axs[2].axis("off")
    # Optionally, add a colorbar:
    # fig.colorbar(im, ax=axs[1])

    plt.tight_layout()

    # Build the output filename.
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_file = os.path.join(args.output_path,args.test_name,f"{base_name}_{args.test_name}.png")
    os.makedirs(os.path.join(args.output_path,args.test_name), exist_ok=True)
    plt.savefig(output_file)
    print(f"Saved overlay to {output_file}")
    plt.close(fig)

def main(args):
    # Use CPU for inference (or uncomment the next line for GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f"Running inference on device: {device}")

    # Load the trained model.
    model = Autoencoder().to(device)
    
    # Load checkpoint with backward compatibility
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Check if checkpoint is in new format (dict with keys) or old format (state_dict only)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format: full checkpoint with optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss: {checkpoint['loss']:.4f}")
    else:
        # Old format: just model state dict (OrderedDict)
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (old format)")
    
    model.eval()

    # Define the transform for model input: resize to (256, 2048), convert to tensor, and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 2048)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # maps [0,1] to [-1,1]
    ])

    # Ensure output folder exists if saving outputs.
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    # Check if image_path is a directory or a single file.
    if os.path.isdir(args.image_path):
        # Get list of image files with common extensions.
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path)
                       if f.lower().endswith(valid_extensions)]
    else:
        image_files = [args.image_path]

    # Process each image.
    for img_path in image_files:
        print(f"Processing image: {img_path}")
        original_img, reconstructed_img, diff = process_image(model, device, img_path, transform)
        blended, anomaly_mask = create_anomaly_overlay(original_img, diff)

        # Option to save a comparison of the original and diff images (both at original resolution)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        if args.output_path:
            if args.save_type == 'compared':
                save_compared(original_img,reconstructed_img, diff, img_path, args)
            else:
                # Alternatively, save the diff image using OpenCV with a colormap.
                # Normalize diff to [0, 255] and apply a "hot" colormap.
                diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                diff_norm = diff_norm.astype(np.uint8)
                colored_diff = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
                output_file = os.path.join(args.output_path, f"{base_name}.jpg")
                cv2.imwrite(output_file, colored_diff)
                
                
        else:
            # If no output path is provided, display the result using OpenCV.
            cv2.imshow("Absolute Difference", diff.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Autoencoder-based Anomaly Detection")
    parser.add_argument("--image_path", type=str, required=False,
                        help="Path to an input image or directory containing images")
    parser.add_argument("--checkpoint_path", type=str, required=False,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional: directory to save the output visualizations")
    parser.add_argument("--test_name", type=str, default=None,
                        help="Optional: name of the test")
    parser.add_argument("--save_type", type=str, default="compared",
                        help="Type of saving: 'compared' for a side-by-side plot or another option for diff only")
    args = parser.parse_args()

    # Manually override some arguments for testing
    args.image_path = r"test_imgs"
    args.checkpoint_path = r"E:\12_AnomalyDetection\0_AUTOENCODER\checkpoints_mod_loss\autoencoder_epoch_18.pth"
    args.output_path = r"localization"
    args.test_name = "AE_mod_loss"
    args.save_type = 'compared'

    main(args)
