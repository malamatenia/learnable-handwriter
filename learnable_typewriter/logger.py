"""Stage 2"""
import os
import PIL
import numpy as np
import torch

from torchvision.utils import make_grid
from learnable_typewriter.model import Model
from learnable_typewriter.utils.generic import nonce, cfg_flatten
from learnable_typewriter.utils.image import img, to_three
from torch.utils.tensorboard import SummaryWriter
import cv2
from PIL import ImageFont, ImageDraw, Image
from os.path import dirname, join
FONT = join(dirname(dirname(__file__)), '.media', 'Junicode.ttf')

def get_size(text, font):
    image = Image.new('RGB', (500, 500), color = (255, 255, 255))
    draw = ImageDraw.Draw(image)
    return draw.textsize(text, font=font)


def text_over_image(image: np.ndarray, text: str, text_color=0):
    h, w = image.shape

    font = ImageFont.truetype(FONT, size=24)
    textsize = get_size('A', font)
    offset = 2*textsize[1]
    img = np.zeros((h + offset, w), dtype=np.uint8)

    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)

    textsize = get_size(text, font)
    text_x, text_y = (w - textsize[0]) // 2, (offset - textsize[1])//2
    draw.text((text_x, text_y), text, font=font, fill=255)
    img = np.array(pil_image)
    
    img[offset:] = (image*255).astype(np.uint8)
    return img/255.0

class Logger(Model):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.__init_tensorboard__()
        if not self.post_load_flag:
            self.__post_init_logger__()

    def __post_init_logger__(self):
        return self.__init_sample_images__()

    def __init_sample_images__(self):
        self.images_to_tsf, self.num_tsf = {True: [], False: []}, self.log_cfg['train']['images']['how_many']
        splits = ['train'] + (['test'] if self.val_flag else [])

        for split in splits:
            for i, dl in enumerate(self.get_dataloader(split=split, batch_size=self.num_tsf, dataset_size=self.num_tsf, remove_crop=True)):
                if self.dataset_kwargs[i].get('split', split) != split:
                    continue
                
                data, alias = next(iter(dl)), self.dataset_kwargs[i]['alias']
                self.images_to_tsf[data['supervised']].append((alias, data))

    def __init_tensorboard__(self, force=False):
        if self.eval and not force:
            self.cfg_flat = None
            self.tensorboard = nonce()
        else:
            tensorboard = self.cfg["training"].get("tensorboard", 'tensorboard')
            log_dir = f"{self.run_dir}/{tensorboard}"

            self.tensorboard = SummaryWriter(log_dir=log_dir)
            self.cfg_flat = cfg_flatten(self.cfg)
            self.keep_steps = []

    def __close_tensorboard__(self):
        self.tensorboard.close()

    def get_metrics(self, split):
        return getattr(self, f'{split}_metrics')

    def log_step(self, split):
        metrics = self.get_metrics(split)
        for k, m in metrics.items():
            msg = f"{split}-{k}-metrics: {m}"
            self.log(msg, eval=True)

    def reset_metrics_test(self):
        for _, m in self.test_metrics.items():
            m.reset()

    def reset_metrics_val(self):
        for _, m in self.val_metrics.items():
            m.reset()

    def reset_metrics_train(self):
        for _, m in self.train_metrics.items():
            m.reset()

    def reset_metrics(self):
        self.reset_metrics_train()
        if self.supervised:
            self.reset_metrics_val()
        self.reset_metrics_test()
        if self.supervised:
            if 'cer_loss_val_' in self.__dict__:
                delattr(self, 'cer_loss_val_')

    def log_val_metrics(self):
        if self.supervised:
            self.eval_reco('val')
            self.log_step('val')
            self.log_tensorboard('val')
        self.eval_reco('test')
        self.log_step('test')
        self.log_tensorboard('test')

    def log_train_metrics(self):
        if self.train_end:
            self.reset_metrics_train()
            self.eval_reco('train')
        self.log_step('train')
        self.log_tensorboard('train')

    @torch.no_grad()
    def log_images(self, header='latest'):
        self.save_prototypes(header)
        self.save_transforms(header)

    def add_image(self, name, x, **kargs):
        x = np.array(x if isinstance(x, PIL.Image.Image) else img(x))

        if len(x.shape) == 2:
            x = np.repeat(x[:, :, None], 3, axis=2)

        self.tensorboard.add_image(name, x, global_step=self.cur_iter, dataformats='HWC', **kargs)

    @torch.no_grad()
    def save_prototypes(self, header): # 00:10 twerk is the new tsifteteli
        masks = self.model.sprites.masks.cpu().numpy()
        if self.supervised:
            masks = torch.stack([torch.from_numpy(text_over_image(masks[i].squeeze(0), self.model.sprite_char[i])).unsqueeze(0) for i in range(masks.shape[0])], dim=0)
        self.save_image_grid(masks, f'{header}/masks', nrow=5)

    @torch.no_grad()
    def save_transforms(self, header):
        self.model.eval()

        for mode, values in self.images_to_tsf.items():
            for alias, images_to_tsf in values:
                decompose = self.decompositor
                obj = decompose(images_to_tsf)
                reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()
                
                obj_al = decompose(images_to_tsf, align=True) # maybe flag that on a future version
                reco_al, seg_al = obj_al['reconstruction'].cpu(), obj_al['segmentation'].cpu()
                
                transformed_imgs = torch.cat([
                    to_three(images_to_tsf['x']).cpu().unsqueeze(0),
                    reco.unsqueeze(0),
                    seg.unsqueeze(0),
                    reco_al.unsqueeze(0),
                    seg_al.unsqueeze(0),
                ], dim=0)
                transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
                self.save_image_grid(transformed_imgs, f'{header}/examples/{mode}/{alias}', nrow=images_to_tsf['x'].size()[0])

    @torch.no_grad()
    def save_image_grid(self, images, title, nrow):
        grid = make_grid(images, nrow=nrow)
        grid = torch.clamp(grid, 0, 1)
        self.add_image(title, grid)

    @property
    def reco_loss_train(self):
        return np.mean([v['reco_loss'].avg for v in self.train_metrics.values()])

    def log_tensorboard(self, split):
        for k, metrics in self.metrics_[split].items():
            losses = list(filter(lambda s: 'loss' in s, metrics.names))
            for l in losses:
                self.tensorboard.add_scalar(f'loss/{k}/{l}/{split}', metrics[l].avg, self.cur_iter)

            losses = list(filter(lambda s: s.startswith('reco'), metrics.names))
            for l in losses:
                self.tensorboard.add_scalar(f'reco/{k}/{l}/{split}', metrics[l].avg, self.cur_iter)

            losses = list(filter(lambda s: s.startswith('time/img'), metrics.names))
            for l in losses:
                self.tensorboard.add_scalar(f'time-per-img/{k}/{l}/{split}', metrics[l].avg, self.cur_iter)

