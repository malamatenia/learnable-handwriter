import numpy as np
import torch
from itertools import chain
from torch import nn
from torch.nn import functional as F

class Selection(nn.Module):
    def __init__(self, dim_enc, dim_sprites, group, logger, num_layers):
        super().__init__()
        self.dim_z = min(dim_enc, dim_sprites)
        logger(f'Selection init with dim_enc={dim_enc}, dim_sprites={dim_sprites} --> dim_z={self.dim_z}')
        self.blank_latent = nn.Parameter(torch.randn((num_layers, self.dim_z)))

        self.linear = nn.Sequential(
            nn.Linear(dim_sprites, self.dim_z),
            nn.LayerNorm(self.dim_z, elementwise_affine=False)
        )

        self.anchors = nn.Sequential(
            nn.Linear(dim_enc, self.dim_z),
            nn.LayerNorm(self.dim_z, elementwise_affine=False)
        )

        self.norm = np.sqrt(self.dim_z)
        self.group, self.log_softmax = group, (self.log_softmax_group if group > 1 else self.log_softmax_)
        self.num_layers = num_layers

    def encoder_params(self):
        return self.anchors.parameters()

    def sprite_params(self):
        return chain(self.linear.parameters(), [self.blank_latent])

    def compute_logits(self, x, sprites):
        latents = sprites.masks_.flat_latents()
        latents = self.linear(latents)
        if self.num_layers == 1:
            latents = torch.cat([latents, self.blank_latent], dim=-1)

        B, C, P = x.size()
        x = x.permute(0, 2, 1).reshape(B*P, C)

        a = self.anchors(x)
        logits = (a @ latents) 

        if self.num_layers > 1:
            # zeros out blanks probabilities that we don't want
            x = x.reshape(B, P, C).reshape(B, P//self.num_layers, self.num_layers, C).reshape(B * (P//self.num_layers), self.num_layers, C)
            # torch.eye(self.num_layers, device=x.device)
            # torch.flip(torch.triu(torch.ones((self.num_layers, self.num_layers), device=x.device)), [1])
            logits_blank = torch.ones((self.num_layers, self.num_layers), device=x.device).unsqueeze(-1).expand(-1, -1, C)
            logits_blank = logits_blank * self.blank_latent.unsqueeze(0)
            # print(x.size(), logits_blank.permute(1, 2, 0).unsqueeze(0).size())
            logits_blank = x.unsqueeze(-1)*logits_blank.permute(1, 2, 0).unsqueeze(0)
            logits_blank = logits_blank.sum(dim=-2)
            logits_blank = logits_blank.reshape(B, P//self.num_layers, self.num_layers, self.num_layers)
            # Should be True False
            #           True False 
            # for L=2
            # torch.count_nonzero(logits_blank[:, :, 0, 0]) == 0, torch.count_nonzero(logits_blank[:, :, 0, 1]) == 0
            # torch.count_nonzero(logits_blank[:, :, 1, 1]) == 0, torch.count_nonzero(logits_blank[:, :, 1, 0]) == 0
            logits_blank = logits_blank.reshape(B, P, self.num_layers).reshape(B*P, self.num_layers)
            logits = torch.cat([logits, logits_blank], dim=-1)

        return logits / self.norm

    def log_softmax_(self, x, logits):
        B, _, L = x.size()
        weights = F.softmax(logits, dim=-1)
        logits = logits.reshape(B, L, -1).permute(1, 0, 2)
        return weights, logits.log_softmax(2)

    def log_softmax_group(self, x, logits):
        B, _, L = x.size()
        weights = F.softmax(logits, dim=-1)
        logits = weights.reshape(B, L, -1).permute(1, 0, 2)
        K = logits.size()[-1]
        all_logits, blank_logits = torch.split(logits, [K-1, 1], dim=-1)
        all_logits = all_logits.view(L, B, (K-1)//self.group, self.group).sum(-1)
        logits = torch.cat([all_logits, blank_logits], dim=-1)
        return weights, logits.log()

    def forward(self, x, sprites):
        """Predicts probabilities for each sprite at each position."""
        B, _, L = x.size()
        logits = self.compute_logits(x, sprites)
        probs, log_probs = self.log_softmax(x, logits)

        output = {'probs': probs, 'logits': logits, 'log_probs': log_probs}
        if not self.training:
            # only one sprite per position
            probs = torch.eye(probs.shape[-1]).to(probs)[probs.argmax(-1)] 
            output['selection'] = probs.reshape(B, L, -1).permute(2, 0, 1)

        # prototypes and masks concatenated
        sprite = torch.cat([
                torch.cat([sprites.prototypes, torch.zeros_like(sprites.prototypes[0]).unsqueeze(0).expand(self.num_layers, -1, -1, -1)], dim=0),
                torch.cat([sprites.masks, torch.zeros_like(sprites.masks[0]).unsqueeze(0).expand(self.num_layers, -1, -1, -1)], dim=0),
            ], dim=1)

        # multiplication of probabilities after the softmax and sprites (masks+colors)
        S = (probs[..., None, None, None] * sprite[None, ...]).sum(1)
        _, C, H, W = S.size()       
        output['S'] = S.reshape(B, L, 4, H, W).permute(1, 0, 2, 3, 4)

        return output
