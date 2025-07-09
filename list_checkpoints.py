#!/usr/bin/env python3
"""
Utility script to list available checkpoints for resuming training.
"""

import os
import glob
import argparse
import torch

def list_checkpoints(checkpoint_dir='./checkpoints'):
    """List all available checkpoint files with their details."""
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return
    
    # Find all .pth files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}'")
        return
    
    print(f"Available checkpoints in '{checkpoint_dir}':")
    print("-" * 60)
    
    for i, checkpoint_path in enumerate(sorted(checkpoint_files), 1):
        filename = os.path.basename(checkpoint_path)
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
        
        # Try to extract epoch number from filename
        epoch_num = "Unknown"
        if 'epoch_' in filename:
            try:
                epoch_num = filename.split('epoch_')[1].split('.')[0]
            except:
                pass
        
        # Try to load checkpoint and extract loss information
        loss_info = "N/A"
        checkpoint_epoch = "N/A"
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if it's the new format with loss information
            if isinstance(checkpoint, dict) and 'loss' in checkpoint:
                loss_info = f"{checkpoint['loss']:.4f}"
                checkpoint_epoch = str(checkpoint['epoch'])
            else:
                loss_info = "Old format (no loss data)"
                
        except Exception as e:
            loss_info = f"Error loading: {str(e)[:30]}..."
        
        print(f"{i:2d}. {filename}")
        print(f"    Epoch (from filename): {epoch_num}")
        print(f"    Epoch (from checkpoint): {checkpoint_epoch}")
        print(f"    Loss: {loss_info}")
        print(f"    Size: {file_size:.1f} MB")
        print(f"    Path: {checkpoint_path}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/checkpoints_smooth_upsample_RUG-STAINS',
                        help="Directory containing checkpoint files")
    
    args = parser.parse_args()
    list_checkpoints(args.checkpoint_dir) 