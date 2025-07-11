import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from models import UNet_BackgroundAware, UNet_AnomalyDetection, UNet_DetailPreserving, Autoencoder_smooth_upsample
from functions import group_nearby_boxes, merge_grouped_boxes, group_and_merge_boxes


def load_model(checkpoint_path, model_type='smooth_upsample'):
    """
    Load a trained autoencoder model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_type: Type of model to load ('smooth_upsample', 'background_aware', 'anomaly_detection', 'detail_preserving')
    
    Returns:
        Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model based on type
    if model_type == 'smooth_upsample':
        model = Autoencoder_smooth_upsample()
    elif model_type == 'background_aware':
        model = UNet_BackgroundAware()
    elif model_type == 'anomaly_detection':
        model = UNet_AnomalyDetection()
    elif model_type == 'detail_preserving':
        model = UNet_DetailPreserving()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] range back to [0, 1] range.
    
    Args:
        tensor: Input tensor in [-1, 1] range
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return (tensor + 1) / 2


def detect_anomalies_and_save_results(img_path, model_path, output_path, 
                                    area_thresh=200, merge_kernel_size=(15, 15), 
                                    p=95, min_spot_area=20, opening_kernel_size=(2, 2), 
                                    crop_padding=15, th_static=180, use_dynamic_threshold=True,
                                    x_distance_threshold=100, y_distance_threshold=10, 
                                    overlap_threshold=0.0):
    """
    Detect anomalies in an image and save results as a single column PNG.
    
    Args:
        img_path: Path to the input image
        model_path: Path to the trained model checkpoint
        output_path: Path to save the output PNG
        area_thresh: Minimum area threshold for defect detection
        merge_kernel_size: Kernel size for morphological closing
        p: Percentile for dynamic threshold calculation
        min_spot_area: Minimum area for individual spots
        opening_kernel_size: Kernel size for opening operation
        crop_padding: Padding around detected regions
        th_static: Static threshold value (used if use_dynamic_threshold=False)
        use_dynamic_threshold: Whether to use dynamic threshold based on percentile
        x_distance_threshold: Distance threshold for merging boxes horizontally
        y_distance_threshold: Distance threshold for merging boxes vertically
        overlap_threshold: Overlap threshold for merging boxes
    """
    
    # Load the original image in color
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    original_shape = original_img.shape[:2]  # (height, width)
    
    # Convert to grayscale for model processing
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray_img)
    
    # Load model
    model = load_model(model_path, 'smooth_upsample')
    
    # Prepare image for model
    target_size = (256, 2048)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # maps [0,1] to [-1,1]
    ])
    
    img_tensor = transform(pil_image).unsqueeze(0)
    
    # Move tensor to the same device as the model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    reconstructed = model(img_tensor)
    
    # Get denormalized images
    original_resized = denormalize(img_tensor.cpu()[0, 0]).detach().numpy()
    reconstructed_img = denormalize(reconstructed.cpu()[0, 0]).detach().numpy()
    
    # Compute the absolute difference
    diff = np.abs((original_resized - reconstructed_img) * 2)
    
    # Create heatmap
    diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    diff_norm = diff_norm.astype(np.uint8)
    heatmap_img = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
    red_channel = heatmap_img[:, :, 2]  # Red channel
    
    # Scale back to original size
    diff_original_size = cv2.resize(diff, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    red_channel_original_size = cv2.resize(red_channel, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Calculate threshold
    if use_dynamic_threshold:
        th = np.percentile(red_channel_original_size.flatten(), p)
    else:
        th = th_static

    # Threshold to create binary mask
    _, mask = cv2.threshold(red_channel_original_size, th, 255, cv2.THRESH_BINARY)
    
    # Remove small spots using opening operation
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel_size)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, merge_kernel_size)
    closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract boxes at original size with additional filtering
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
            
        # Additional filtering: remove very narrow or very tall regions (likely noise)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / max(1, min(w, h))  # Avoid division by zero
        
        # Filter out very narrow or very tall regions (aspect ratio > 10)
        if aspect_ratio > 10:
            continue
            
        # Filter out very small regions that might be noise
        if area < min_spot_area:
            continue
            
        boxes.append((x, y, w, h))
    
    # Merge boxes
    final_boxes = group_and_merge_boxes(boxes, x_distance_threshold=x_distance_threshold, 
                                       y_distance_threshold=y_distance_threshold, 
                                       overlap_threshold=overlap_threshold)
    
    # Create visualization images
    # 1. Difference image in original size
    diff_viz = cv2.applyColorMap(
        cv2.normalize(diff_original_size, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_HOT
    )
    
    # 2. Mask after threshold and cleaning in original size with detected areas highlighted
    mask_viz = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    
    # Draw boxes on the mask to show detected areas
    for box in boxes:
        x, y, w, h = box
        # Apply padding but don't go out of image bounds
        x0 = max(x - crop_padding, 0)
        y0 = max(y - crop_padding, 0)
        x1 = min(x + w + crop_padding, original_shape[1])
        y1 = min(y + h + crop_padding, original_shape[0])
        cv2.rectangle(mask_viz, (x0, y0), (x1, y1), (0, 255, 0), 4)  # Green boxes on mask
    
    # 3. Original image with individual boxes (thicker)
    img_with_boxes = original_img.copy()
    for box in boxes:
        x, y, w, h = box
        # Apply padding but don't go out of image bounds
        x0 = max(x - crop_padding, 0)
        y0 = max(y - crop_padding, 0)
        x1 = min(x + w + crop_padding, original_shape[1])
        y1 = min(y + h + crop_padding, original_shape[0])
        cv2.rectangle(img_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 4)  # Thicker green boxes
    
    # 4. Original image with merged boxes (thicker)
    img_with_merged_boxes = original_img.copy()
    for box in final_boxes:
        x, y, w, h = box
        # Apply padding but don't go out of image bounds
        x0 = max(x - crop_padding, 0)
        y0 = max(y - crop_padding, 0)
        x1 = min(x + w + crop_padding, original_shape[1])
        y1 = min(y + h + crop_padding, original_shape[0])
        cv2.rectangle(img_with_merged_boxes, (x0, y0), (x1, y1), (0, 0, 255), 4)  # Thicker red boxes
    
    # Create visualization for opened mask (before closing)
    mask_opened_viz = cv2.cvtColor(mask_opened, cv2.COLOR_GRAY2BGR)
    
    # Create single column visualization
    # Get the width of the original image
    img_width = original_shape[1]
    
    # Create a single column image with all 5 visualizations
    total_height = original_shape[0] * 5
    combined_img = np.zeros((total_height, img_width, 3), dtype=np.uint8)
    
    # Place images in the combined image
    y_offset = 0
    combined_img[y_offset:y_offset + original_shape[0]] = diff_viz
    y_offset += original_shape[0]
    
    combined_img[y_offset:y_offset + original_shape[0]] = mask_opened_viz
    y_offset += original_shape[0]
    
    combined_img[y_offset:y_offset + original_shape[0]] = mask_viz
    y_offset += original_shape[0]
    
    combined_img[y_offset:y_offset + original_shape[0]] = img_with_boxes
    y_offset += original_shape[0]
    
    combined_img[y_offset:y_offset + original_shape[0]] = img_with_merged_boxes
    
    # Save the result
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = f"{output_path}/{base_name}_anomaly_results.png"
    
    cv2.imwrite(output_path, combined_img)
    print(f"Results saved to: {output_path}")
    print(f"Detected {len(boxes)} individual regions and {len(final_boxes)} merged regions")
    
    return combined_img, boxes, final_boxes


def process_folder(input_folder, model_path, output_folder, **kwargs):
    """
    Process all images in a folder and save anomaly detection results.
    
    Args:
        input_folder: Path to folder containing images
        model_path: Path to the trained model checkpoint
        output_folder: Path to save the output PNGs
        **kwargs: Additional parameters to pass to detect_anomalies_and_save_results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        try:
            print(f"Processing {i}/{len(image_files)}: {img_path.name}")
            combined_img, boxes, final_boxes = detect_anomalies_and_save_results(
                str(img_path), model_path, output_folder, **kwargs
            )
            print(f"  ✓ Detected {len(boxes)} individual regions and {len(final_boxes)} merged regions")
            print(f"Results saved to: {output_folder}/{img_path.name}_anomaly_results.png")
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Results saved to: {output_folder}")


if __name__ == "__main__":
    # Example usage for single image
    # img_path = r"E:\12_AnomalyDetection\0_AUTOENCODER\test_imgs\221013_0912320_00022_2_0.jpg"
    # model_path = r"E:\12_AnomalyDetection\0_AUTOENCODER\checkpoints/checkpoints_smooth_upsample_RUG-STAINS\autoencoder_epoch_40.pth"
    # output_path = r"E:\12_AnomalyDetection\0_AUTOENCODER\0_anomaly_output"
    # combined_img, boxes, final_boxes = detect_anomalies_and_save_results(img_path, model_path, output_path)
    
    # Example usage for folder processing
    input_folder = r"E:\12_AnomalyDetection\0_AUTOENCODER\sliver_test"
    model_path = r"E:\12_AnomalyDetection\0_AUTOENCODER\checkpoints/checkpoints_smooth_upsample_RUG-STAINS\autoencoder_epoch_40.pth"
    output_folder = r"E:\12_AnomalyDetection\0_AUTOENCODER\sliver_test\sliver_result"
    
    # Example parameters - you can modify these as needed
    params = {
        'area_thresh': 200,                    # Minimum area for defect detection
        'merge_kernel_size': (15, 15),         # Kernel size for morphological closing
        'p': 95,                               # Percentile for dynamic threshold
        'min_spot_area': 20,                   # Minimum area for individual spots
        'opening_kernel_size': (2, 2),         # Kernel size for opening operation
        'crop_padding': 15,                    # Padding around detected regions
        'th_static': 180,                      # Static threshold value
        'use_dynamic_threshold': True,         # Use dynamic threshold based on percentile
        'x_distance_threshold': 100,           # Distance threshold for merging boxes horizontally
        'y_distance_threshold': 10,            # Distance threshold for merging boxes vertically
        'overlap_threshold': 0.0               # Overlap threshold for merging boxes
    }
    
    # Process all images in the folder with custom parameters
    process_folder(input_folder, model_path, output_folder, **params)