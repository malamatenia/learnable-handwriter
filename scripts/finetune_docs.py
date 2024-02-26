import os
import sys
import json
import PIL
import numpy as np
import plotly.express as px
import pandas as pd
import torch

from tqdm import tqdm
from os.path import dirname, abspath, join
from collections import defaultdict
from finetune_scripts import check_patch, get_loader, eval_ft

from torchvision.transforms import ToPILImage
LIB_PATH = join(dirname(dirname(abspath(__file__))))
sys.path.append(LIB_PATH)
from learnable_typewriter.utils.loading import load_pretrained_model
from torchvision.utils import make_grid

def cpu_clone(sd):
    return {k: torch.clone(v.cpu()) for k, v in sd.items()}

def save_finetuned_checkpoint(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)

def load_finetuned_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

def locate_changed_keys(a, b):
    return [k for k in a.keys() if k in b.keys() and not torch.equal(a[k], b[k])]

def get_documents(path, split, filter=None):
    with open(path, 'r') as f:
        annotation = json.load(f)

    document_keys = [k.split('_')[0] for k, v in annotation.items() if v['split'] == split or split == 'all']

    split_dict = defaultdict(set)
    for k, v in annotation.items():
        if v['split'] == split or split == 'all':
            split_dict[k.split('_')[0]].add(v['split'])

    for k, v in split_dict.items():
        split_dict[k] = ('all' if len(v) == 2 else list(v)[0])

    print(f"Document_keys: {document_keys}")

    documents = sorted(set(document_keys))  # Use sorted to maintain order
    print(f"Documents: {documents}")
    assert len(documents)

    document_counts = {doc: document_keys.count(doc) for doc in documents}

    for doc, count in document_counts.items():
        print(f"Document {doc}: {count} keys")

    return list(zip(documents, [split_dict[doc] for doc in documents]))

def plot_sprites(trainer, drr, invert_sprites):
    os.makedirs(drr, exist_ok=True)
    masks = trainer.model.sprites.masks
    if invert_sprites:
        masks = 1 - masks
    
    for k in range(len(trainer.model.sprites)):
        ToPILImage()(masks[k]).save(join(drr, f'{k}.png'))
    ToPILImage()(make_grid(masks, nrow=4)).save(join(drr, f'grid.png'))
    ToPILImage()(make_grid(masks, nrow=len(trainer.model.sprites))).save(join(drr, f'grid-1l.png'))

def finetune(trainer, max_steps, log_every, save_sprites_dir, reconstructions_path, save_model_dir, invert_sprites, split):
    parent_state_dict = cpu_clone(trainer.model.state_dict())
    trainer.model.encoder.eval()

    i = 0
    train_loss, losses = [], []
    pbar = tqdm(desc="Training", total=max_steps, leave=True)
    loader = get_loader(trainer, split)
    while i < max_steps:
        for x in loader:
            trainer.optimizer.zero_grad()
            loss = trainer.model(x)['loss']
            losses.append(loss.item())
            loss.backward()
            trainer.optimizer.step()

            if i%log_every == 0:
                train_loss.append((i, np.mean(losses)))
                plot_sprites(trainer, join(save_sprites_dir, f'step_{i}'), invert_sprites=invert_sprites)
                eval_ft(trainer, join(reconstructions_path, f'step_{i}.png'), split)
                losses = []

            if i == 10:
                print('learned keys (state_dict comparison)', locate_changed_keys(parent_state_dict, cpu_clone(trainer.model.state_dict()))) #differences from initial state_dict until now

            i += 1
            pbar.update(1)

            if i >= max_steps:
                break

    save_finetuned_checkpoint(trainer.model, save_model_dir)
    print('[on save] learned keys (state_dict comparison)', locate_changed_keys(torch.load(save_model_dir, map_location='cpu'), cpu_clone(trainer.model.state_dict())))
    plot_sprites(trainer, join(save_sprites_dir, 'final'), invert_sprites=invert_sprites)
    eval_ft(trainer, join(reconstructions_path, f'final.png'), split)
    fig = px.line(pd.DataFrame(train_loss, columns=['step', 'training-loss']), x="step", y="training-loss")
    fig.write_image(join(reconstructions_path, f'loss.png'))

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
    documents = get_documents(args.annotation_file, args.split, args.filter)
    print(f"Filtered Documents size: {len(documents)}")
    os.makedirs(join(args.output_path, args.sprites_path), exist_ok=True)
    pbar = tqdm(documents)
    os.makedirs(join(args.output_path, 'models'), exist_ok=True)
    for document, split in documents:
        pbar.set_description(f"Processing {document}")

        trainer = load_pretrained_model(path=args.model_path, device=str(args.device), conf=make_optimizer_conf(args))

        from omegaconf import OmegaConf
        config_file = join(args.output_path, 'config.yaml')
        with open(config_file, 'w') as f:
            f.write(OmegaConf.to_yaml(trainer.cfg))

        transcribe_file = join(args.output_path, args.sprites_path, 'transcribe.json')
        if not os.path.isfile(transcribe_file):
            with open(transcribe_file, 'w') as f:
                json.dump(trainer.model.transcribe, f, indent=4)

        plot_sprites(trainer, join(args.output_path, 'baseline'), invert_sprites=args.invert_sprites)
        for k in trainer.dataset_kwargs:
            k['filter_by_name'] = document
            k['path'] = args.data_path
            k['annotation_path'] = args.annotation_file

        # TODO uncomment this line when the patch is ready
        check_patch(trainer, split, args, join(args.output_path, document)) 
        trainer.val_loader, trainer.test_loader = [], []
        finetune(trainer, max_steps=args.max_steps, log_every=args.log_every, save_sprites_dir=join(args.output_path, document, args.sprites_path), reconstructions_path=join(args.output_path, document, args.reconstructions_path), save_model_dir=join(args.output_path, 'models', f'{document}.pth'), invert_sprites=args.invert_sprites, split=split)
        torch.cuda.empty_cache()

    baseline = PIL.Image.open(join(args.output_path, 'baseline', f'grid-1l.png'))
    png_list = [stack([baseline for _ in range(len(documents))])] #(len(args.script))])]
    png_list += [stack([baseline] + [PIL.Image.open(join(args.output_path, document, args.sprites_path, str(i).zfill(3), f'grid-1l.png')) for document in documents]) for i in range(args.max_steps)]
    png_list[0].save(join(args.output_path, 'progress.gif'), save_all=True, duration=len(png_list)*0.1, append_images=png_list[1:])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Helper script to finetune Textualis on a specific set of documents or script and generate figures.')
    parser.add_argument('-i', "--model_path", type=str, required=True)
    parser.add_argument('-o', "--output_path", required=True, default=None)
    parser.add_argument('-f', "--filter", nargs="+", default=None)
    parser.add_argument('-r', "--reconstructions_path", default='reco', type=str)
    parser.add_argument('-s', "--sprites_path", default='sprites', type=str)
    parser.add_argument('-d', "--data_path", type=str, required=True)
    parser.add_argument('-a', "--annotation_file", required=True, default=join(LIB_PATH, 'datasets/south_north_textualis_mask/annotation.json'), type=str)
    parser.add_argument("--max_steps", required=False, default=2000, type=int)
    parser.add_argument("--log_every", required=False, default=500, type=int)
    parser.add_argument("--mode", choices=["all", "sprites", "g_theta"], default='g_theta')
    parser.add_argument("--invert_sprites", action='store_true')
    parser.add_argument("--enable_experimental", action='store_true')
    parser.add_argument("--device", default=0, type=str) 
    parser.add_argument('--split', choices=["all", "val", "train"], required=True, help='Name of the script (e.g., Northern_Textualis, Southern_Textualis)')
    args = parser.parse_args()

    run(args)
