"""Stage 2"""
import os
import PIL
import numpy as np
import torch

from torchvision.utils import make_grid
from learnable_typewriter.model import Model
from learnable_typewriter.utils.generic import nonce, cfg_flatten, add_nest
from learnable_typewriter.utils.image import img, to_three
import wandb


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
        self.__init_wandb__()
        if not self.post_load_flag:
            self.__post_init_logger__()

    def __post_init_logger__(self):
        return self.__init_sample_images__()

    def __init_sample_images__(self):
        self.images_to_tsf, self.num_tsf = {True: [], False: []}, self.log_cfg['train']['images']['how_many']
        splits = ['train'] + (['val'] if self.val_flag else []) #we changed that from +test to +val because we split the dataset to train+val without test 18/10

        for split in splits:
            for i, dl in enumerate(self.get_dataloader(split=split, batch_size=self.num_tsf, dataset_size=self.num_tsf, remove_crop=True)):
                if self.dataset_kwargs[i].get('split', split) != split:
                    continue

                data, alias = next(iter(dl)), self.dataset_kwargs[i]['alias']
                self.images_to_tsf[data['supervised']].append((alias, data))

    def __init_wandb__(self, force=False):
        if self.eval and not force:
            self.cfg_flat = None
            self.wandb = nonce()
        else:
            self.wandb = wandb.init(
                project='learnable-scriber',
                name=self.cfg["tag"],
                config=cfg_flatten(self.cfg)
            )
            self.log_wandb_ = {}

    def __close_wandb__(self):
        self.wandb.finish()

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
            self.log_wandb('val')
        self.eval_reco('test')
        self.log_step('test')
        self.log_wandb('test')

    def log_train_metrics(self):
        if self.train_end:
            self.reset_metrics_train()
            self.eval_reco('train')
        self.log_step('train')
        self.log_wandb('train')

    @torch.no_grad()
    def log_images(self, header='latest'):
        self.save_prototypes(header)
        self.save_transforms(header)

    def add_image(self, name, x, **kargs):
        x = np.array(x if isinstance(x, PIL.Image.Image) else img(x))

        if len(x.shape) == 2:
            x = np.repeat(x[:, :, None], 3, axis=2)

        add_nest(self.log_wandb_, name, wandb.Image(x))

    @torch.no_grad()
    def save_prototypes(self, header): # 00:10 twerk is the new tsifteteli
        masks = self.model.sprites.masks.cpu().numpy() #type is numpy.ndarray
        if self.supervised:
            # prints in alphanumerical order
            masks = torch.stack([torch.from_numpy(text_over_image(masks[i].squeeze(0), self.model.sprite_char[i])).unsqueeze(0) for i in range(masks.shape[0])], dim=0) 
        self.save_image_grid(masks, f'masks/{header}', nrow=5)

        if self.supervised:
            masks = self.model.sprites.masks.cpu().numpy()
            # Get the character occurrences order
            sorted_indices = sorted(self.character_occurrences, key=lambda k: self.character_occurrences[k], reverse=True)
            # Process and stack the sorted masks
            masks = torch.stack([torch.from_numpy(text_over_image(masks[i].squeeze(0), self.model.sprite_char[i])).unsqueeze(0) for i in sorted_indices], dim=0)
            self.save_image_grid(masks, f'masks-ordered/{header}', nrow=5)

    @torch.no_grad()
    def save_transforms(self, header):
        self.model.eval()

        for mode, values in self.images_to_tsf.items():
            for alias, images_to_tsf in values:
                decompose = self.decompositor
                obj = decompose(images_to_tsf)
                reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()

                objs = [
                    to_three(images_to_tsf['x']).cpu().unsqueeze(0),
                    reco.unsqueeze(0),
                    seg.unsqueeze(0),
                ]
                try:
                    obj_al = decompose(images_to_tsf, align=True) # maybe flag that on a future version
                    reco_al, seg_al = obj_al['reconstruction'].cpu(), obj_al['segmentation'].cpu()
                    objs.append(reco_al.unsqueeze(0))
                    objs.append(seg_al.unsqueeze(0))
                except AttributeError:
                    import warnings; warnings.warn('Imputer load fail... Ignoring decompose.')
                
                transformed_imgs = torch.cat(objs, dim=0)
                images = [
                    wandb.Image(
                        img(
                            torch.clamp(torch.cat([
                                transformed_imgs[i][j]
                                for i in range(transformed_imgs.shape[0])
                            ], dim=1), 0, 1)
                        )
                    ) for j in range(transformed_imgs.shape[1])]
                modet = ('supervised' if mode else 'unsupervised')
                add_nest(self.log_wandb_, f'{alias}/examples/{header}/{modet}', images)

    @torch.no_grad()
    def save_image_grid(self, images, title, nrow):
        grid = make_grid(images, nrow=nrow)
        grid = torch.clamp(grid, 0, 1)
        self.add_image(title, grid)

    @property
    def reco_loss_train(self):
        return np.mean([v['reco_loss'].avg for v in self.train_metrics.values()])

    def log_wandb(self, split):
        for k, metrics in self.metrics_[split].items():
            for t in ['loss', 'reco', 'time/img']:
                losses = list(filter(lambda s: t in s, metrics.names))
                for l in losses:
                    lt = l.replace('_train', '').replace('_test', '').replace('_val', '')
                    if metrics[l].count > 0:
                        add_nest(self.log_wandb_, f'{k}/{lt}/{split}', metrics[l].avg)

    def push_wandb(self):
        if len(self.log_wandb_):
            wandb.log(self.log_wandb_, step=self.cur_iter)
            self.log_wandb_ = {}