import warnings 
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import open_dict

from learnable_typewriter.utils.image import to_three

from learnable_typewriter.typewriter.typewriter.window import Window
from learnable_typewriter.typewriter.typewriter.encoder import Encoder
from learnable_typewriter.typewriter.typewriter.selection import Selection

from learnable_typewriter.typewriter.typewriter.sprites.sprites import Sprites
from learnable_typewriter.typewriter.typewriter.sprites.background import Background
from learnable_typewriter.typewriter.typewriter.sprites.transformations import Transformation

from learnable_typewriter.typewriter.typewriter.compositor import LayeredCompositor
from learnable_typewriter.typewriter.optim.loss import Loss

delta = lambda x: None


class LearnableTypewriter(nn.Module):
    def __init__(self, cfg, transcribe, logger=delta):
        super().__init__()

        if transcribe is not None:
            with open_dict(cfg):
                cfg.sprites.n = len(transcribe)
        
        self.cfg = cfg
        self.log = logger
        self.transcribe = transcribe
        
        self.canvas_size = cfg['transformation']['canvas_size']
        self.log(f'canvas_size = {self.canvas_size}')

        self.encoder = Encoder(self.cfg['encoder'])
        self.window = Window(self.encoder, self.cfg['window'])
        self.sprites = Sprites(self.cfg['sprites'], logger)
        self.background = Background(self.cfg['background'])
        self.selection = Selection(self.encoder.out_ch, self.sprites.masks_.latent_dim, self.sprites.per_character, logger, num_layers=self.encoder.L)
        if self.background:
            self.background_transformation = Transformation(self, 1, self.cfg['transformation']['background'], background=True)
            self.register_buffer('beta_bkg', torch.cat([torch.eye(2, 2), torch.zeros((2, 1))], dim=-1).unsqueeze(0))

        self.layer_transformation = Transformation(self, 1, self.cfg['transformation']['layer'])
        self.loss = Loss(self, cfg['loss'])

        self.compositor = LayeredCompositor(self)
        self.debug = False

    @property
    def mode(self):
        return self.sprites.masks_.mode

    @property
    def device(self):
        return self.sprites.masks_.proto.device

    @property
    def masks(self):
        return self.sprites.masks

    @property
    def prototypes(self):
        return self.sprites.prototypes

    @property
    def is_layer_tsf_id(self):
        if hasattr(self, 'layer_transformation'):
            return self.layer_transformation.only_id_activated
        else:
            return False

    def forward(self, x, return_params=False):
        # Predict the spatial transformation parameters
        y, xr = {}, self.predict_cell_per_cell(x, return_params=return_params, return_masks_frg=self.loss.reg_overlap(x))
        loss = self.loss(x, xr)

        y['loss'] = loss['total']
        for k in loss.keys():
            if k.endswith('loss'):
                y[k] = loss[k]

        if return_params:
            y['params'] = xr['params']
        
        return y

    def predict_parameters(self, x, features):
        tsf_bkgs_param = None
        if self.background:
            tsf_bkgs_param = torch.sigmoid(self.background_transformation.predict_parameters(x, features['background']).permute(1, 2, 0, 3))

        tsf_layers_params = self.layer_transformation.predict_parameters(x, features['sprites']).squeeze(0)
        return tsf_layers_params, tsf_bkgs_param

    def transform_background(self, color, size):
        beta = self.beta_bkg.expand(size[0], -1, -1)
        grid = F.affine_grid(beta, size, align_corners=False)
        out = F.grid_sample(color, grid, mode='bilinear', padding_mode='border', align_corners=False)[0]
        return out

    def transform_sprites_p(self, sprites_p, params_layers_p):
        # We apply the transformation on the sprites
        B, _, h, w = sprites_p.size()

        # Layer Transformation
        tsf_sprites_p = self.layer_transformation.apply_parameters(sprites_p.unsqueeze(0), params_layers_p.unsqueeze(0))
        tsf_sprites_p = tsf_sprites_p.view(B, -1, self.encoder.H, self.window.w)

        tsf_layers, tsf_masks = torch.split(tsf_sprites_p, [self.encoder.C, 1], dim=1) #The torch.split function takes as its second argument a list of integers that specifies the size of each output tensor along the given dimension. In this case, the list [self.encoder.C, 1] specifies that tsf_layers should have C channels and tsf_masks should have a single channel.
        return tsf_layers, tsf_masks
    
    def predict_cell_per_cell(self, x, return_masks_frg=False, return_params=False): #The method predict_cell_per_cell takes an input image and applies the learned transformation to each "cell" of the image separately, allowing the model to learn spatial transformations that can be applied locally to specific regions of the input.
        img = x['x'] = x['x'].to(self.device) #The input image is first passed through an encoder to extract features
        B, C, H, W = img.size() #save BCHW in variables
        C = min(C, 3) 
        features = self.encoder(img)

        tsf_layers_params, tsf_bkgs_param = self.predict_parameters(img, features) #predicts transformation parameters for the input image and its background 
        self.transform_layers_ = tsf_layers_params

        # transform backgrounds
        if self.background:
            tsf_bkgs = self.transform_background(tsf_bkgs_param, (B, C, H, W)) #If the background flag is set to True, the background of the input image is transformed using the predicted background parameters.

        # select which sprites to place
        selection = self.selection(features['sprites'], self.sprites) # selects which sprites to place

        # transform sprites
        all_tsf_layers, all_tsf_masks = self.transform_sprites(selection['S'], tsf_layers_params) # transformed using the predicted transformation parameters
        composed = self.compositor(to_three(img), tsf_bkgs, all_tsf_layers, all_tsf_masks) #compose the transformed sprites and background onto the input image

        output = {
            'reconstruction': composed['cur_img'],
            'tsf_layers': all_tsf_layers,
            'tsf_masks': all_tsf_masks, 
            'tsf_bkgs': tsf_bkgs,
            'probs': selection['probs'],
            'logits' : selection['logits'],
            'log_probs': selection['log_probs']
        }

        if return_params:
            output['params'] = {'sprites': tsf_layers_params, 'background': tsf_bkgs_param}

        if return_masks_frg:
            output['cur_masks'] = composed['cur_mask']
            output['cur_foregrounds'] = composed['cur_foreground']

        if not self.selection.training:
            output['selection'] = selection['selection']

        return output 

    def step(self):
        pass

    def transform_sprites(self, sprites, tsf_layers_params):
        n_cells = tsf_layers_params.size(-1) #an integer representing the size of the tsf_layers_params tensor along its last dimension = features or representations

        tsf_sprites, tsf_masks = [], []
        for p in range(n_cells):
            tsf_layers_params_p = tsf_layers_params[:, :, p] # For each cell index p, tsf_layers_params_p is defined as a slice of tsf_layers_params along the last dimension
            tsf = self.transform_sprites_p(sprites[p], tsf_layers_params_p)

            tsf_sprites.append(tsf[0])
            tsf_masks.append(tsf[1])

        if hasattr(self, 'noise'):
            del self.noise

        return tsf_sprites, tsf_masks
    
    @property
    def n_prototypes(self):
        return len(self.sprites)

    @property
    def empty_sprite_id(self):
        return len(self.sprites)

    def set_optimizer(self, opt):
        self.optimizer = opt
        if hasattr(self, 'layer_transformation'):
            self.layer_transformation.set_optimizer(opt)
        if self.background:
            self.background_transformation.set_optimizer(opt)

    def load_state_dict(self, state_dict): # need to fix
        state = self.state_dict()
        
        diff = (state_dict.keys() | state.keys()) - (state_dict.keys() & state.keys())
        if len(diff) > 0:
            diff_amb = state_dict.keys() - self.state_dict()
            if len(diff_amb):
                warnings.warn(f'load_state_dict: The following keys were found in loaded dict but not in self.state_dict():\n{diff_amb}')

            diff_bma = self.state_dict() - state_dict.keys()
            if len(diff_amb):
                warnings.warn(f'load_state_dict: The following keys were found in self.state_dict() dict but not in loaded dict:\n{diff_bma}')

        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                state[name].copy_(param)

    def prototypes_parameters(self):
        params = [self.sprites.masks_.parameters()]
        if self.background:
            params.append([self.background.background_])
        return chain(*params)

    def transformation_parameters(self):
        params = [self.layer_transformation.parameters()]
        if self.background:
            params.append(self.background_transformation.parameters())
        return chain(*params)

    @torch.no_grad()
    def process_batch_transcriptions(self, y):
        '''Transforms batch transcriptions to 2d tensor
        Outputs:
            - transcriptions : 2d tensor (B, len_max) that contains for each batch the transcription in sprites ids
            - lengths_gt : list, contains the true length of transcription for each instance of the batch       
        '''
        len_max = max([len(k) for k in y])
        transcriptions_padded = torch.full(size=(len(y), len_max), fill_value=self.empty_sprite_id)  
        lengths_gt = torch.full(size=(len(y), ), fill_value=len_max, device=self.device, dtype=torch.int64)

        for b in range(len(y)):
            transcriptions_padded[b,:len(y[b])] = torch.tensor(y[b])
            lengths_gt[b] = len(y[b])

        return transcriptions_padded, lengths_gt

    @torch.no_grad()
    def true_width_pos(self, x, widths, n_cells):
        '''Outputs for each instance its true length in terms of positions'''
        B, _, _, w_max = x.size()
        widths_gt_pos = torch.full(size=(B,), fill_value=n_cells)
        available_instances = torch.ones(B, dtype=torch.bool)

        for p in range(n_cells):
            ws = self.window.global_index(p, w_max)[0]
            eliminated_instances = torch.where((widths < ws) & available_instances)[0]
            widths_gt_pos[eliminated_instances] = p - 1
            available_instances[eliminated_instances] = False 

        return widths_gt_pos.to(self.device)

    def selection_head(self, x):
        img = x['x'] = x['x'].to(self.device)
        features = self.encoder(img)
        return self.selection(features['sprites'], self.sprites)
