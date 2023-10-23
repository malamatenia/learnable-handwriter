from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from learnable_typewriter.typewriter.typewriter.sprites.utils import create_mlp_with_conv1d, copy_with_noise

########################
#  ABC
########################

class _AbstractTransformationModule(nn.Module):
    __metaclass__ = ABCMeta
    identity_module = False

    @abstractmethod
    def transform(self, x, beta):
        return self._transform(x, beta)

    def __bool__(self):
        return not self.identity_module

    def load_with_noise(self, module, noise_scale):
        if bool(self):
            self.load_state_dict(module.state_dict())
            self.mlp[-1].bias.data.copy_(copy_with_noise(module.mlp[-1].bias, noise_scale))

    @property
    def dim_parameters(self):
        try:
            dim_parameters = self.mlp[-1].out_features
        except AttributeError as e:
            dim_parameters = self.mlp[-1].out_channels
        return dim_parameters


########################
#   Modules 
########################

class IdentityModule(_AbstractTransformationModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.identity_module = True
 
    def predict_parameters(self, x, *args, **kargs):
        return x

    def transform(self, x, *args, **kwargs):
        return x

    def load_with_noise(self, module, noise_scale):
        pass


class ColorModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.color_ch = kwargs['color_channels'] 
        n_layers = kwargs['n_hidden_layers'] 
        n_hidden_units = kwargs['n_hidden_units'] 

        # MLP
        self.mlp = create_mlp_with_conv1d(in_channels, self.color_ch, n_hidden_units, n_layers) #an MultiLayeredPerceptron is created with 1D convolutional layers
        self.mlp[-1].weight.data.zero_() #The weights and biases of the last layer of the MLP are initialized to zero
        self.mlp[-1].bias.data.zero_()

        # Identity transformation parameters
        self.register_buffer('identity', torch.eye(self.color_ch, self.color_ch)) #torch.eye = Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

    def predict_parameters(self, x):
        return self.mlp(x) # predicted parameters are returned

    def transform(self, x, beta):
        if x.size(1) == 2 or x.size(1) > 3: #If x has 2 channels or more than 3 channels, the split is based on self.color_ch
            x, mask = torch.split(x, [self.color_ch, x.size(1) - self.color_ch], dim=1) #The input x is split into two parts: x and mask.
        else:
            mask = None

        if x.size(1) == 1:  #If x has only 1 channel, it is expanded to 3 channels.
            x = x.expand(-1, 3, -1, -1)

        weight = beta.view(-1, self.color_ch, 1) 
        weight = weight.expand(-1, -1, self.color_ch) * self.identity + self.identity #The predicted parameters (beta) are reshaped and multiplied by the identity matrix, creating a weight matrix (weight).

        output = torch.einsum('bij, bjkl -> bikl', weight, x) #The transformation is applied using Einstein summation (torch.einsum) on the dimensions of weight and x
        output = torch.sigmoid(output) 
        output = torch.clamp(output, 0, 1) ##Sigmoid activation and clamping ensure that the output values are between 0 and 1
        if mask is not None:
            output = torch.cat([output, mask], dim=1) #If mask is not None, it is concatenated with the output along the channel dimension.
        
        return output #a transformed tensor

class PositionModule(_AbstractTransformationModule):
    def __init__(self, in_channels, canvas_size, **kwargs):
        super().__init__()
        self.padding_mode = kwargs['padding_mode']
        self.parametrization = kwargs['parametrization']

        if self.parametrization not in ['exp', 'sinh']:
            raise ValueError(self.parametrization)

        self.Hs, self.Ws = kwargs['sprite_size']
        self.H, self.W = int(canvas_size[0]), int(canvas_size[1])

        # MLP Init
        n_layers = kwargs['n_hidden_layers']
        n_hidden_units = kwargs['n_hidden_units']

        self.mlp = create_mlp_with_conv1d(in_channels, 3, n_hidden_units, n_layers) #for s and t
        self.mlp[-1].weight.data.zero_() #The weights and biases of the last layer of the MLP are initialized to zero
        self.mlp[-1].bias.data.zero_()

        # Spatial constraint
        self.register_buffer('t', torch.Tensor([kwargs['max_x'], kwargs['max_y']]).unsqueeze(0).unsqueeze(-1))

        # Identity transformation parameters
        eye = torch.eye(2, 2)
        eye[0, 0], eye[1, 1] = self.W/self.Ws, self.H/self.Hs
        self.register_buffer('eye', eye)

    def predict_parameters(self, x):
        beta = self.mlp(x) #beta is the transformation parameters

        s, t = beta.split([1, 2], dim=1) #it's split into scale and translation (shift) factors

        if self.parametrization == 'exp': #scale is modified based on the specified parametrization (exp or sinh)
            s = torch.exp(s)
        elif self.parametrization == 'sinh': 
            s = torch.sinh(s)

        t = torch.clamp(t, min=-1.0, max=1.0)*self.t #the translation is clamped to be between -1.0 and 1.0

        return torch.cat([s, t], dim=1) #concatenation

    def transform(self, x, beta):
        s, t = beta.split([1, 2], dim=1)
        scale = s[..., None].expand(-1, 2, 2) * self.eye #A scaling matrix (scale) is created based on the scale parameter.
        beta = torch.cat([scale, t.unsqueeze(2)], dim=2) #A new set of transformation parameters (beta) is created by combining the scaling matrix and translation
 
        # grid is a batch of affine matrices
        grid = F.affine_grid(beta, (x.size(0), x.size(1), self.H, self.W), align_corners=False)

        # The size-2 vector grid[n, h, w] specifies input pixel locations x and y with n = batch size * nb_sprites + empty sprite
        return F.grid_sample(x, grid, mode='bilinear', padding_mode=self.padding_mode, align_corners=False)
