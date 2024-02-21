import os
import sys
import json
import PIL
from PIL import Image
from tqdm import trange, tqdm
from os.path import dirname, abspath, join

import torch
import numpy as np
from torchvision.transforms import ToPILImage
LIB_PATH = join(dirname(dirname(abspath(__file__))))
sys.path.append(LIB_PATH)
from learnable_typewriter.utils.loading import load_pretrained_model
from torchvision.utils import make_grid
import plotly.express as px
import pandas as pd

def cpu_clone(sd):
    return {k: torch.clone(v.cpu()) for k, v in sd.items()}

def save_finetuned_checkpoint(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)

def load_finetuned_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

def locate_changed_keys(a, b):
    return [k for k in a.keys() if k in b.keys() and not torch.equal(a[k], b[k])]

def get_documents_by_script(path, split_filter, script_filter=None):
    with open(path, 'r') as f:
        annotation = json.load(f)

    documents = set([k for k, v in annotation.items() if v['split'] == split_filter and v['script'] == script_filter])
    assert len(documents)
    return sorted(documents)

def eval(trainer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for images_to_tsf in trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=False)[0]:
        decompose = trainer.decompositor
        obj = decompose(images_to_tsf)
        reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()
        nrow = images_to_tsf['x'].size()[0]
        transformed_imgs = torch.cat([images_to_tsf['x'][:, :3].cpu().unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
        transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
        grid = make_grid(transformed_imgs, nrow=nrow)
        ToPILImage()(torch.clamp(grid, 0, 1)).save(path)
        return

def plot_sprites(trainer, drr, invert_sprites):
    os.makedirs(drr, exist_ok=True)
    masks = trainer.model.sprites.masks
    if invert_sprites:
        masks = 1 - masks

    for k in range(len(trainer.model.sprites)):
        ToPILImage()(masks[k]).save(join(drr, f'{k}.png'))
    ToPILImage()(make_grid(masks, nrow=4)).save(join(drr, f'grid.png'))
    ToPILImage()(make_grid(masks, nrow=len(trainer.model.sprites))).save(join(drr, f'grid-1l.png'))

def get_loader(trainer, split):
    if split == 'all':
        from itertools import chain
        return chain(
            trainer.get_dataloader(split='train', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True)[0],
            trainer.get_dataloader(split='val', batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True)[0]
        )
    else:
        return trainer.get_dataloader(split=split, batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True)[0]

def finetune(trainer, max_steps, save_sprites_dir, reconstructions_path, save_model_dir, invert_sprites, split):
    parent_state_dict = cpu_clone(trainer.model.state_dict())
    i, pbar = 0, tqdm(desc="Training", total=max_steps, leave=True)
    while i < max_steps:
        for x in get_loader(trainer, split):
            trainer.optimizer.zero_grad()
            loss = trainer.model(x)['loss']
            loss.backward()
            trainer.optimizer.step()

            if i == 10:
                print('learned keys (state_dict comparison)', locate_changed_keys(parent_state_dict, cpu_clone(trainer.model.state_dict()))) #differences from initial state_dict until now

            i += 1
            pbar.update(1)

            if i >= max_steps:
                break

    save_finetuned_checkpoint(trainer.model, save_model_dir)
    print('[on save] learned keys (state_dict comparison)', locate_changed_keys(torch.load(save_model_dir, map_location='cpu'), cpu_clone(trainer.model.state_dict())))
    plot_sprites(trainer, join(save_sprites_dir, 'final'), invert_sprites=invert_sprites)
    eval(trainer, join(reconstructions_path, f'final.png'))

def stack(imgs):
    dst = PIL.Image.new('RGB', (imgs[0].width, len(imgs)*imgs[0].height))
    for i, img in enumerate(imgs):
        dst.paste(img, (0, i*imgs[0].height))
    return dst

def make_optimizer_conf(args):
    conf = {}
    if args.mode == "sprites":
        conf["training"] = {
                "optimizer": {
                    "lr": 0,
                    "prototypes": {   
                        "lr": 0.0001
                    },
                    "encoder": {'weight_decay': 0}
                }
            }
    elif args.mode == "g_theta":
        conf["training"] = {
                "optimizer": {
                    "lr": 0.0001,
                    "finetune": "g_theta"                    
                }
        }
    return conf    

def run(args):
    os.makedirs(join(args.output_path, args.sprites_path), exist_ok=True)
    pbar = tqdm(args.script)
    os.makedirs(join(args.output_path, 'models'), exist_ok=True)
    for document in pbar:
        pbar.set_description(f"Processing {document}")
        trainer = load_pretrained_model(path=args.model_path, device=str(args.device), conf=make_optimizer_conf(args))
        transcribe_file = join(args.output_path, args.sprites_path, 'transcribe.json')
        if not os.path.isfile(transcribe_file):
            with open(transcribe_file, 'w') as f:
                json.dump(trainer.model.transcribe, f, indent=4)

        plot_sprites(trainer, join(args.output_path, 'baseline'), invert_sprites=args.invert_sprites)
        for k in trainer.dataset_kwargs:
            k['script'] = document
            k['path'] = args.data_path
            k['annotation_path'] = args.annotation_file

        trainer.val_loader, trainer.test_loader = [], []
        finetune(trainer, max_steps=args.max_steps, save_sprites_dir=join(args.output_path, document, args.sprites_path), save_model_dir=join(args.output_path, 'models', f'{document}.pth'), reconstructions_path=join(args.output_path, document, args.reconstructions_path), invert_sprites=args.invert_sprites, split=args.split)
        torch.cuda.empty_cache()
    
    baseline = PIL.Image.open(join(args.output_path, 'baseline', f'grid-1l.png'))
    png_list = [stack([baseline for _ in range(len(args.script))])]
    png_list += [stack([baseline] + [PIL.Image.open(join(args.output_path, document, args.sprites_path, str(i).zfill(3), f'grid-1l.png')) for document in args.script]) for i in range(args.max_epochs)]
    png_list[0].save(join(args.output_path, 'progress.gif'), save_all=True, duration=len(png_list)*0.1, append_images=png_list[1:])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Helper script to finetune Textualis on a specific set of documents or script and generate figures.')
    parser.add_argument('-i', "--model_path", type=str, required=True)
    parser.add_argument('-o', "--output_path", required=True, default=None)
    parser.add_argument('-r', "--reconstructions_path", default='reco', type=str)
    parser.add_argument('-s', "--sprites_path", default='sprites', type=str)
    parser.add_argument('-d', "--data_path", type=str, required=True)
    parser.add_argument('-a', "--annotation_file", required=True, default=join(LIB_PATH, 'datasets/south_north_textualis_mask/annotation.json'), type=str)
    parser.add_argument("--max_steps", required=False, default=30000, type=int)
    parser.add_argument("--mode", choices=["all", "sprites", "g_theta"], default='g_theta')
    parser.add_argument("--invert_sprites", action='store_true')
    parser.add_argument("--device", default=0, type=str)
    
    # New argument for script name - can put to False if not required as filter for the finetuning
    parser.add_argument('--script', nargs="+", required=True, help='Name of the script (e.g., Northern_Textualis, Southern_Textualis)')
    parser.add_argument('--split', choices=["all", "val", "train"], required=True, help='Name of the script (e.g., Northern_Textualis, Southern_Textualis)')

    args = parser.parse_args()
    run(args)