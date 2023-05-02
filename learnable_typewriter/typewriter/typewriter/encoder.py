import numpy as np
import torch
from torch import nn, zeros
from torch.nn import functional as F
from learnable_typewriter.utils.image import to_three
from learnable_typewriter.typewriter.typewriter.mini_resnet import get_resnet_model

DOWNSCALE_FACTOR = {'resnet32': 4, 'resnet20': 4, 'resnet14': 4, 'resnet8': 4, 'default': 5} 


def create_positional_encoding(max_len, d_model):
    # same size with input matrix (for adding with input matrix)
    encoding = torch.zeros(max_len, d_model) #sprites per position x dimensions 
    encoding.requires_grad = False  # we don't need to compute gradient

    pos = torch.arange(0, max_len)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, d_model, step=2).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    return encoding


def gaussian_kernel(size, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    ax = np.linspace(-(size[1] - 1)/ 2., (size[1]-1)/2., size[1])
    ay = np.linspace(-(size[0] - 1)/ 2., (size[0]-1)/2., size[0])

    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)

    return torch.as_tensor(kernel, dtype=dtype).unsqueeze(0).unsqueeze(0)


class GaussianPool(nn.Module):
    def __init__(self, size, stride):
        super().__init__()
        self.kernel = gaussian_kernel(size)
        self.stride = stride

    def forward(self, x):
        self.kernel = self.kernel.to(x.device)
        kernel = self.kernel.expand(x.size()[1], -1, -1, -1)
        return F.conv2d(x, weight=kernel, stride=self.stride, padding=0, groups=x.size()[1])


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.C, self.H = cfg['C'], cfg['H']  #encoder.C is the number of input channels to the encoder module. It is used to initialize a ResNet model with in_channels=self.C. Later on in the code, the output from the encoder is processed to extract features, and self.out_ch is the number of output channels from the encoder.

        resnet = get_resnet_model(cfg['name'])(in_channels=self.C)
        seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.layer1, resnet.layer2, resnet.layer3]

        # The resnet32 layer3 divide each dimension by 4 => 50 for an image in [40, 200]
        self.feature_downscale_factor = DOWNSCALE_FACTOR[cfg['name']]
        self.pool_h = self.H // self.feature_downscale_factor
        self.pool_w = cfg['pooling_on_width']

        seq.append(GaussianPool((self.pool_h, self.pool_w), stride=(1, self.pool_w)))

        self.encoder = nn.Sequential(*seq)
        self.out_ch = self.encoder(zeros(1, 3, self.H, 2*self.H)).size()[1]
        self.layer_norm = nn.LayerNorm(self.out_ch, elementwise_affine=False)

    def forward(self, x): #we pass from 4D to 3D and we permute the order as to have Batch size, P(positions), d(features)
        x = to_three(x)
        x = self.encoder(x)
        x = self.layer_norm(x.squeeze(2).permute(0, 2, 1)).permute(0, 2, 1)
        return x
