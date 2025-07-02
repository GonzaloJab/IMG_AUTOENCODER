import os
import argparse
from PIL import Image

# from model import Autoencoder
from model import Autoencoder

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pytorch_msssim import ssim

from tqdm import tqdm  # progress bar

# Custom dataset for loading images from a directory
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Filter common image extensions.
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Open image and convert to grayscale ("L" mode)
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image


def SSIMLoss(y_pred, y_true):
    # Here, data_range is set to 2 because our images are normalized to [-1, 1]
    return 1 - ssim(y_pred, y_true, data_range=2.0, size_average=True)

def total_variation(recon):
    dy = torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :])
    dx = torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1])
    return dx.mean() + dy.mean()

def train(args):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Define the transform: resize to 256x2048, convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 2048)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # (x - 0.5) / 0.5 -> [-1,1]
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(root_dir=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Instantiate model, optimizer, and send model to device
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if specified
    start_epoch = 1
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Loading checkpoint from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            
            # Check if checkpoint is in new format (dict with keys) or old format (state_dict only)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format: full checkpoint with optimizer state
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch} (loss: {checkpoint['loss']:.4f})")
            else:
                # Old format: just model state dict (OrderedDict)
                model.load_state_dict(checkpoint)
                
                # Try to extract epoch number from filename for logging purposes
                filename = os.path.basename(args.resume_from)
                if 'epoch_' in filename:
                    try:
                        start_epoch = int(filename.split('epoch_')[1].split('.')[0]) + 1
                        print(f"Resuming training from epoch {start_epoch} (old checkpoint format)")
                    except:
                        print("Could not determine epoch number from filename, starting from epoch 1")
                else:
                    print("Resuming training from epoch 1 (old checkpoint format)")
        else:
            print(f"Warning: Checkpoint file {args.resume_from} not found. Starting training from scratch.")

    # Training Loop
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running_loss = 0.0

        # Wrap the dataloader with tqdm for a progress bar.
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{args.epochs}]", leave=False)
        for batch_idx, inputs in enumerate(progress_bar):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = SSIMLoss(outputs, inputs)
            
            # Calculate losses
            ssim_loss = SSIMLoss(outputs, inputs)
            tv_loss = total_variation(outputs)
            
            # Combine losses with weight for total variation
            loss = ssim_loss + args.tv_weight * tv_loss
            
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(
                ssim_loss=f"{ssim_loss.item():.4f}", 
                tv_loss=f"{tv_loss.item():.4f}",
                total_loss=f"{loss.item():.4f}"
            )

        avg_loss = running_loss / len(dataloader)
        tqdm.write(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f} (TV weight: {args.tv_weight})")

        # Save model checkpoint every few epochs or at the end
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"autoencoder_epoch_{epoch}.pth")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            # Save both model state dict and optimizer state dict
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'tv_weight': args.tv_weight,

            }
            torch.save(checkpoint, checkpoint_path)
            tqdm.write(f"Model checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Autoencoder for Anomaly Detection")
    parser.add_argument('--data_dir', type=str, default=r'E:\0_DATASETS\NONE',
                        help="Path to the directory containing training images")
    
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='./checkpoints_mod_loss',
                        help="Directory where model checkpoints will be saved")
    
    parser.add_argument('--batch_size', type=int, 
                        default=2,
                        help="Batch size for training")
    
    parser.add_argument('--epochs', type=int, 
                        default=20,
                        help="Number of epochs to train")
    
    parser.add_argument('--learning_rate', type=float, 
                        default=0.0002,
                        help="Learning rate for optimizer")
    
    parser.add_argument('--num_workers', type=int, 
                        default=4,
                        help="Number of worker threads for data loading")
    
    parser.add_argument('--log_interval', type=int, 
                        default=4,
                        help="How many batches to wait before logging training status")
    
    parser.add_argument('--save_interval', type=int, 
                        default=2,
                        help="How many epochs to wait before saving a checkpoint")
    
    parser.add_argument('--resume_from', type=str, 
                        default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument('--tv_weight', type=float, 
                        default=0.001,
                        help="Weight for total variation loss")
      
      
    #parse args
    args = parser.parse_args()
    # args.resume_from = r"checkpoints_mod_loss\autoencoder_epoch_20.pth"
    
    train(args)