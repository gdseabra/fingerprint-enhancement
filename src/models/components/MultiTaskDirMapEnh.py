""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.components.ResUNet import ResUNet
from models.components.UNet import UNet

class MultiTaskDirMapEnh(nn.Module):
    def __init__(self, in_ch=1, out_ch=90, ndim=2, chs: tuple[int, ...] = (64, 128, 256, 512, 1024)):
        super(MultiTaskDirMapEnh, self).__init__()
        
        self.dirmap_net = UNet(in_ch=2, out_ch=90, ndim=ndim, chs=chs)
        self.enhancer_net = ResUNet(in_ch=1, out_ch=2, ndim=ndim)

    def forward(self, x):
        
        # enh_input = torch.concat([out_dirmap, x], axis=1)
        out_enh = self.enhancer_net(x)
        out_dirmap = self.dirmap_net(out_enh)

        return out_dirmap, out_enh




if __name__ == '__main__':
    model         =  MultiTaskDirMapEnh()

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    summary(model, (1, 256, 256))