import torch
from torch import nn

class CTC(torch.nn.Module):
    def __init__(self, blank, zero_infinity=True, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.loss = torch.nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def __call__(self, log_probs, y, input_lengths, target_lengths):
        return self.loss(log_probs, y, input_lengths.to(log_probs.device), target_lengths.to(log_probs.device))

class Loss(object):
    def __init__(self, model, cfg):
        self.model = model
        self._l2 = nn.MSELoss(reduction='none')
        self.ctc_factor = cfg['ctc_factor']
        if self.ctc_factor > 0:
            self.ctc = CTC(blank=model.blank)
        self.__exp_unsup_regularizers_init__(cfg)

    def __exp_unsup_regularizers_init__(self, cfg):
        self.overlap_, self.freq_, self.sparse_ = cfg.get('overlap', 0), cfg.get('frequency', 0), cfg.get('sparse', 0)
        self.model.log(f'Overlapping Penalization loss ' + (f'ON {self.overlap_} > 0' if self.overlap_ else 'OFF'))
        self.model.log(f'Frequency loss ' + (f'ON {self.freq_} > 0' if self.freq_ else 'OFF'))
        self.model.log(f'Sparse loss ' + (f'ON {self.sparse_} > 0' if self.sparse_ else 'OFF'))

    def reg_ctc(self, x):
        return self.ctc_factor > 0 and x['supervised']

    def reg_overlap(self, x):
        return self.overlap_ > 0 and not x['supervised']

    def reg_blank_sparsity(self, x):
        return self.sparse_ > 0 and not x['supervised']

    def reg_frequency(self, x):
        return self.freq_ > 0 and not x['supervised']

    def penalization(self, cur_masks):
        cur_masks = cur_masks.abs()
        pen, n_cells = None, cur_masks.shape[0]

        for i in range(n_cells):
            mask_i = cur_masks[i]
            for j in range(i + 1, n_cells):
                mask_j = cur_masks[j]
                elem = (mask_i * mask_j).flatten(1).sum(1)
                pen = (pen + elem if pen is not None else elem)

        return pen

    def overlap_penalization(self, x):
        return self.overlap_ * self.penalization(x['cur_masks']).mean()

    @property
    def supervised(self):
        return self.model.supervised

    def l2_(self, pred, gt):
        masked = False
        if gt.size(1) == 4:
            mask = gt[:, -1].unsqueeze(1)
            gt = gt[:, :-1]
            masked = True

        elif len(gt.size()) == 3:
            gt = gt.unsqueeze(1)

        loss = self._l2(pred, gt)
        if masked:
            #mask_ = mask.sum(dim=-1).sum(dim=-1).sum(dim=-1)
            #mask_ = torch.where(mask > 0, torch.ones_like(mask), mask)
            loss = (loss * mask) #/mask_

        return gt,loss

    def l2(self, gt, pred):
        if gt['cropped']:
            return self.l2_(pred, gt['x'])[1].mean()
        else:
            x, loss = self.l2_(pred, gt['x'])
            mask = self.get_mask_width(x, torch.tensor(gt['w']))
            return (loss*mask).sum(-1).mean(2).mean() 

    def get_mask_width(self, gt, widths):
        mask_widths = torch.zeros_like(gt)
        for b in range(len(gt)):
            mask_widths[b, :, :, :widths[b]] = 1/widths[b]
        return mask_widths

    def unsup_reg(self, loss, output, gt, pred):
        if self.reg_overlap(gt):
           overpen = self.overlap_penalization(pred)
           output['reg_overlap'] = overpen.detach().item()
           loss = loss + overpen

        if self.reg_blank_sparsity(gt):
            sparse = self.sparse_ * pred['w'][..., -1].mean()
            output['reg_sparse'] = sparse.detach().item()
            loss = loss + sparse

        if self.reg_frequency(gt):
            freq = 1 - torch.clamp(pred['w'].mean(0), max=self.freq_/pred['w'].size()[-1]).sum(-1)/self.freq_
            output['reg_freq'] = freq.detach().item()
            loss = loss + freq

        return loss

    def sup_reg(self, loss, output, gt, pred):
        if self.reg_ctc(gt):
            transcriptions_padded, true_lengths = self.model.process_batch_transcriptions(gt['base'])
            n_cells = self.model.transform_layers_.size(-1)
            true_widths_pos = self.model.true_width_pos(gt['x'], torch.Tensor(gt['w']), n_cells)
            log_probs = pred['log_probs']
            ctc_loss = self.ctc_factor*self.ctc(log_probs, transcriptions_padded, true_widths_pos, true_lengths)
            output['ctc_loss'] = ctc_loss.detach().item()
            loss = loss + ctc_loss

        return loss

    def reco(self, output, gt, pred):
        loss = self.l2(gt, pred['reconstruction'])
        output['reco_loss'] = loss.detach().item()
        return loss

    def __call__(self, gt, pred):
        output = {}
        loss = self.reco(output, gt, pred)
        loss = self.sup_reg(loss, output, gt, pred)
        loss = self.unsup_reg(loss, output, gt, pred)
        output['total'] = loss
        return output
