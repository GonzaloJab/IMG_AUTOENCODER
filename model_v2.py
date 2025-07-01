import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
