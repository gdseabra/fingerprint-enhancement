import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic Convolutional Block: Conv -> Norm -> Activation"""
    def __init__(self, in_channels, out_channels, norm_layer=nn.GroupNorm, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = norm_layer(1, out_channels)  # 1 group for small channel sizes
        self.act = activation(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ndim=2, chs=(16, 32, 64)):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(chs[2] + chs[1], chs[1])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(chs[1] + chs[0], chs[0])

        self.final_conv = nn.Conv2d(chs[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Decoder
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.final_conv(x)
