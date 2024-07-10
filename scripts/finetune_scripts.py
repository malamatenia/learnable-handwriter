import os
import sys
import json
import PIL
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import dirname, abspath, join, isfile

import torch
LIB_PATH = join(dirname(dirname(abspath(__file__))))
sys.path.append(LIB_PATH)
from learnable_typewriter.utils.loading import load_pretrained_model
import plotly.express as px

from finetune_utils import check_patch, get_loader, eval_ft, save_finetuned_checkpoint, locate_changed_keys, cpu_clone, plot_sprites, get_parser, stack, make_optimizer_conf

def finetune(trainer, max_steps, log_every, save_sprites_dir, reconstructions_path, save_model_dir, invert_sprites, split):
    parent_state_dict = cpu_clone(trainer.model.state_dict())
    train_loss, losses = [], []
    i, j, pbar = 0, 0, tqdm(desc="Training", total=max_steps, leave=True)
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
                plot_sprites(trainer, join(save_sprites_dir, str(i).zfill(3)), invert_sprites=invert_sprites)
                eval_ft(trainer, join(reconstructions_path, str(i).zfill(3) + '.png'), split)
                losses = []

                # measure the difference in weights for the changed keys
                # print('learned keys (state_dict diff-absolu)', {k: torch.norm(parent_state_dict[k] - cpu_clone(trainer.model.state_dict())[k]) for k in locate_changed_keys(parent_state_dict, cpu_clone(trainer.model.state_dict()))})


            if i == 10:
                # differences from initial state_dict until now
                print('learned keys (state_dict comparison)', locate_changed_keys(parent_state_dict, cpu_clone(trainer.model.state_dict())))

    
            i += 1
            pbar.update(1)

            if i >= max_steps:
                break

            pbar.set_postfix({'epoch': j, 'loss': loss.item()})

        j += 1
    
    save_finetuned_checkpoint(trainer.model, save_model_dir)
    print('[on save] learned keys (state_dict comparison)', locate_changed_keys(torch.load(save_model_dir, map_location='cpu'), cpu_clone(trainer.model.state_dict())))
    plot_sprites(trainer, join(save_sprites_dir, 'final'), invert_sprites=invert_sprites)
    eval_ft(trainer, join(reconstructions_path, f'final.png'), split)
    fig = px.line(pd.DataFrame(train_loss, columns=['step', 'training-loss']), x="step", y="training-loss")
    fig.write_image(join(reconstructions_path, f'loss.png'))

def run(args):
    os.makedirs(join(args.output_path, args.sprites_path), exist_ok=True)
    pbar = tqdm(args.script)
    
    for document in pbar:
        pbar.set_description(f"Processing {document}")

        # Skip if document already exists
        document_path = join(args.output_path, document)
        if isfile(join(document_path, 'model.pth')): # Check in document's directory
            print(f"Document {document} already exists. Skipping...")
            continue

        trainer = load_pretrained_model(path=args.model_path, device=str(args.device), conf=make_optimizer_conf(args))

        os.makedirs(join(args.output_path, args.sprites_path, 'transcribe'), exist_ok=True)
        transcribe_file = join(args.output_path, args.sprites_path, 'transcribe', f'{document}.json')
        if not os.path.isfile(transcribe_file):
            with open(transcribe_file, 'w') as f:
                json.dump(trainer.model.transcribe, f, indent=4)

        from omegaconf import OmegaConf
        config_file = join(args.output_path, 'config.yaml')
        with open(config_file, 'w') as f:
            f.write(OmegaConf.to_yaml(trainer.cfg))

        plot_sprites(trainer, join(args.output_path, 'baseline'), invert_sprites=args.invert_sprites)
        for k in trainer.dataset_kwargs:
            k['script'] = document
            k['path'] = args.data_path
            k['annotation_path'] = args.annotation_file

        check_patch(trainer, args.split, args, document_path)
        trainer.val_loader, trainer.test_loader = [], []
        
        # Update save_model_dir to save 'model.pth' in the document's directory
        finetune(trainer, max_steps=args.max_steps, log_every=args.log_every, 
                 save_sprites_dir=join(document_path, args.sprites_path), 
                 save_model_dir=join(document_path, 'model.pth'), 
                 reconstructions_path=join(document_path, args.reconstructions_path), 
                 invert_sprites=args.invert_sprites, split=args.split)
                 
        plot_sprites(trainer, join(args.output_path, args.sprites_path), 
                     invert_sprites=args.invert_sprites, save_individual=False, grid_pref=document)
        torch.cuda.empty_cache()

        try:
            # Initialize png_list with baseline image
            png_list = [PIL.Image.open(join(args.output_path, 'baseline', f'grid-1l.png'))]

            # Append images for each step up to args.max_steps
            for i in range(args.max_steps):
                png_file = join(document_path, args.sprites_path, str(i).zfill(3), f'grid-1l.png')
                if os.path.isfile(png_file):
                    png_list.append(PIL.Image.open(png_file))

            # Save the GIF using png_list
            png_list[0].save(join(document_path, 'progress.gif'), save_all=True, duration=len(png_list) * 0.1, append_images=png_list[1:])
        except Exception as e:
            print(f"Error creating gif: {e}")


if __name__ == "__main__":
    parser = get_parser('scripts')
    parser.add_argument('--script', nargs="+", required=True, help='Name of the script (e.g., Northern_Textualis, Southern_Textualis)')
    args = parser.parse_args()
    run(args)
