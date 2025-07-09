import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim  # pip install pytorch-msssim

class Autoencoder_simple(nn.Module):
    def __init__(self):
        super(Autoencoder_simple, self).__init__()
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
class Autoencoder_simple_v2(nn.Module):
    def __init__(self):
        super(Autoencoder_simple_v2, self).__init__()
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
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=1)
        
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

class Autoencoder_smooth_upsample(nn.Module):
    def __init__(self):
        super(Autoencoder_smooth_upsample, self).__init__()
        # Encoder (unchanged)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bn5   = nn.BatchNorm2d(512)
        
        # Decoder: Upsample + Conv to restore both dims
        self.up1   = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6   = nn.BatchNorm2d(512)
        
        self.up2   = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn7   = nn.BatchNorm2d(256)
        
        self.up3   = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn8   = nn.BatchNorm2d(128)
        
        self.up4   = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn9   = nn.BatchNorm2d(64)
        
        # Final conv back to 1 channel
        self.conv10 = nn.Conv2d(64, 1, 3, padding=1)
        
        # Activations
        self.act = nn.LeakyReLU(0.3)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        
        # Decoder
        x = self.up1(x)
        x = self.act(self.bn6(self.conv6(x)))
        
        x = self.up2(x)
        x = self.act(self.bn7(self.conv7(x)))
        
        x = self.up3(x)
        x = self.act(self.bn8(self.conv8(x)))
        
        x = self.up4(x)
        x = self.act(self.bn9(self.conv9(x)))
        
        x = self.tanh(self.conv10(x))
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder (Contracting Path)
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Decoder (Expanding Path)
        self.up6_1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=1)  # 512 = 256(up) + 256(skip)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.up7_1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)  # 256 = 128(up) + 128(skip)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        
        self.up8_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=1)   # 128 = 64(up) + 64(skip)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        
        self.up9_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=1)    # 64 = 32(up) + 32(skip)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(32)
        
        # Final output layer
        self.conv10 = nn.Conv2d(32, 1, 1)  # 1x1 conv for final output
        
        # Activations
        self.act = nn.LeakyReLU(0.3)
        self.tanh = nn.Tanh()
        
        # Max pooling for downsampling
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # Encoder (Contracting Path)
        # Level 1
        x1 = self.act(self.bn1(self.conv1_2(self.act(self.conv1_1(x)))))
        
        # Level 2
        x2 = self.pool(x1)
        x2 = self.act(self.bn2(self.conv2_2(self.act(self.conv2_1(x2)))))
        
        # Level 3
        x3 = self.pool(x2)
        x3 = self.act(self.bn3(self.conv3_2(self.act(self.conv3_1(x3)))))
        
        # Level 4
        x4 = self.pool(x3)
        x4 = self.act(self.bn4(self.conv4_2(self.act(self.conv4_1(x4)))))
        
        # Level 5 (Bottleneck)
        x5 = self.pool(x4)
        x5 = self.act(self.bn5(self.conv5_2(self.act(self.conv5_1(x5)))))
        
        # Decoder (Expanding Path)
        # Level 6
        up6 = self.up6_1(x5)
        # Skip connection: concatenate with x4
        up6 = torch.cat([up6, x4], dim=1)
        x6 = self.act(self.bn6(self.conv6_2(self.act(self.conv6_1(up6)))))
        
        # Level 7
        up7 = self.up7_1(x6)
        # Skip connection: concatenate with x3
        up7 = torch.cat([up7, x3], dim=1)
        x7 = self.act(self.bn7(self.conv7_2(self.act(self.conv7_1(up7)))))
        
        # Level 8
        up8 = self.up8_1(x7)
        # Skip connection: concatenate with x2
        up8 = torch.cat([up8, x2], dim=1)
        x8 = self.act(self.bn8(self.conv8_2(self.act(self.conv8_1(up8)))))
        
        # Level 9
        up9 = self.up9_1(x8)
        # Skip connection: concatenate with x1
        up9 = torch.cat([up9, x1], dim=1)
        x9 = self.act(self.bn9(self.conv9_2(self.act(self.conv9_1(up9)))))
        
        # Final output
        x10 = self.tanh(self.conv10(x9))
        return x10