import numpy as np
import torch
from itertools import chain
from torch import nn
from torch.nn import functional as F
from learnable_typewriter.typewriter.typewriter.alignment.best_alignment import best_alignment

def best_path(x, log_probs, model, zero_infinity=False):
    transcriptions_padded, true_lengths = model.process_batch_transcriptions(x['base'])
    n_cells = model.transform_layers_.size(-1)
    true_widths_pos = model.true_width_pos(x['x'], torch.Tensor(x['w']), n_cells)
    alignment = best_alignment(log_probs, transcriptions_padded, true_widths_pos.to(log_probs.device), true_lengths.to(log_probs.device), blank=model.blank, zero_infinity=zero_infinity)
    output = alignment.transpose(1, 0).contiguous()
    return output

class Selection(nn.Module):
    def __init__(self, dim_enc, dim_sprites, group, factoring, logger):
        super().__init__()
        self.dim_z = min(dim_enc, dim_sprites)
        logger(f'Selection init with dim_enc={dim_enc}, dim_sprites={dim_sprites} --> dim_z={self.dim_z}')
        self.blank_latent = nn.Parameter(torch.randn((1, self.dim_z)))

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
        assert factoring is not None, "for none factoring not implemented yet"
        factoring = torch.from_numpy(factoring).to(self.blank_latent.dtype)
        factoring = torch.cat([factoring, torch.zeros((1, factoring.shape[1]), dtype=factoring.dtype)], dim=0)
        factoring = torch.cat([factoring, torch.zeros((factoring.shape[0], 1), dtype=factoring.dtype)], dim=1)
        factoring[-1, -1] = 1
        self.register_buffer('factoring', factoring) # |A| times |c|

    def encoder_params(self):
        return self.anchors.parameters()

    def sprite_params(self):
        return chain(self.linear.parameters(), [self.blank_latent])

    def compute_logits(self, x, sprites):
        latents = sprites.masks_.flat_latents()
        latents = self.linear(latents)
        latents = torch.cat([latents, self.blank_latent], dim=0)
        latents = self.factoring.matmul(latents).transpose(1,0).contiguous()

        B, C, P = x.size()
        x = x.permute(0, 2, 1).reshape(B*P, C)

        a = self.anchors(x)
        logits = (a @ latents)

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

    def forward(self, x, sprites, gt_align=None, model=None):
        """Predicts probabilities for each sprite at each position."""
        B, _, L = x.size()
        logits = self.compute_logits(x, sprites)
        probs, log_probs = self.log_softmax(x, logits)

        output = {'probs': probs, 'logits': logits, 'log_probs': log_probs}
        if gt_align is not None:
            # Implement alignment to refine probas
            one_hot = best_path(gt_align, log_probs, model, zero_infinity=True).reshape(B*L, -1).float()

            # print('A+', ''.join([model.transcribe[t.item()] for t in torch.unique_consecutive(one_hot.argmax(dim=-1)[0]) if t != one_hot.size(-1)]))
            if self.training:
                probs = probs * one_hot
            else:
                probs = one_hot

            # print('A', ''.join([model.transcribe[t.item()] for t in torch.unique_consecutive(probs.reshape(B, L, -1).argmax(dim=-1)[0]) if t != probs.size(-1)]))
        elif not self.training:
            probs = torch.eye(probs.shape[-1]).to(probs)[probs.argmax(-1)]
            # print('B', ''.join([model.transcribe[t.item()] for t in torch.unique_consecutive(probs.reshape(B, L, -1).argmax(dim=-1)[0]) if t != probs.size(-1)]))

        if not self.training:
            output['selection'] = probs.reshape(B, L, -1).permute(2, 0, 1)

        probs = probs.matmul(self.factoring)        

        # prototypes and masks concatenated
        sprite = torch.cat([
                torch.cat([sprites.prototypes, torch.zeros_like(sprites.prototypes[0]).unsqueeze(0)], dim=0),
                torch.cat([sprites.masks, torch.zeros_like(sprites.masks[0]).unsqueeze(0)], dim=0),
            ], dim=1)

        # multiplication of probabilities after the softmax and sprites (masks+colors)
        S = (probs[..., None, None, None] * sprite[None, ...]).sum(1)
        _, C, H, W = S.size()       
        output['S'] = S.reshape(B, L, 4, H, W).permute(1, 0, 2, 3, 4)

        return output
