import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim  # pip install pytorch-msssim

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5   = nn.BatchNorm2d(512)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6     = nn.BatchNorm2d(512)
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7     = nn.BatchNorm2d(256)
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn8     = nn.BatchNorm2d(128)
        
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn9     = nn.BatchNorm2d(64)
        
        # Final layer: stride=1 and tanh activation
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        # Activations
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)  # matching a higher slope (e.g., 0.3) similar to Keras
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        
        # Decoder
        x = self.leaky_relu(self.bn6(self.deconv1(x)))
        x = self.leaky_relu(self.bn7(self.deconv2(x)))
        x = self.leaky_relu(self.bn8(self.deconv3(x)))
        x = self.leaky_relu(self.bn9(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x

def SSIMLoss(y_pred, y_true):
    """
    Computes the SSIM loss.
    The pytorch_msssim.ssim function returns a similarity measure in [0, 1],
    so we subtract it from 1 to get a loss (where 0 is perfect).
    """
    return 1 - ssim(y_pred, y_true, data_range=1.0, size_average=True)

# Instantiate the model and optimizer
# model = Autoencoder()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Example training loop snippet:
# Assuming you have a DataLoader that returns batches of images shaped (batch, 1, 256, 2048)
# for epoch in range(num_epochs):
#     for inputs in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = SSIMLoss(outputs, inputs)
#         loss.backward()
#         optimizer.step()
