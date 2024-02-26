import os
import sys
import json
import PIL
import unicodedata
from os.path import dirname, abspath, join
import argparse

import torch
from torchvision.transforms import ToPILImage
LIB_PATH = join(dirname(dirname(abspath(__file__))))
sys.path.append(LIB_PATH)
from torchvision.utils import make_grid

def log_transcribe(tag, values):
    print(tag.capitalize() + ':', json.dumps(list(values)))

def check_patch(trainer, split, args, save_dir):
    try:
        # trainer.get_dataset(trainer.dataset_kwargs[0], (None if split == 'all' else split), force=True)
        get_loader(trainer, split)
        log_transcribe('current dictionary', trainer.transcribe_dataset.values())

        # if the dictionary differs this error will pop. Then we will need to patch the trainer.
    except KeyError as e:
        if args.enable_experimental:
            log_transcribe('previous dictionary', trainer.transcribe_dataset.values())

            assert trainer.model.sprites.per_character == 1, 'This patch only works for per_character == 1'

            # TODO find which type of exception this is and catch that and only that.
            print('Patching trainer because of dictionary mismatch:', e)

            # step 1: extract all metadata related to old dictionary and delete it from trainer
            matching_old = {v: k for k, v in trainer.transcribe_dataset.items()}
            alphabet_old = set(matching_old.keys())
            unique_combining_old = {v: i for i, v in enumerate(make_factoring(alphabet_old))}
            delattr(trainer, 'transcribe_dataset')

            # step 2: initialize dataset in trainer.
            dataset = trainer.get_dataset(trainer.dataset_kwargs[0], (None if split == 'all' else split), force=True)
            log_transcribe('new dictionary', trainer.transcribe_dataset.values())

            factoring = dataset.factoring[0]
            matching = {v: k for k, v in trainer.transcribe_dataset.items()}
            alphabet = set(matching.keys())
            unique_combining = {v: i for i, v in enumerate(make_factoring(alphabet))}

            # step 3: Compute matching between old and new alphabet
            # find the the keys that are in both unique combining and unique combining old and connect the new index to old index
            common_keys = set(unique_combining.keys()).intersection(set(unique_combining_old.keys()))
            print('New keys:', set(unique_combining.keys()).difference(set(unique_combining_old.keys())))
            print('Discarded keys:', set(unique_combining_old.keys()).difference(set(unique_combining.keys())))
            new_to_old_common_sprites = {unique_combining[k]: unique_combining_old[k] for k in common_keys}

            # step 4: shift the order of latents in g_theta to obey the new dictionary and reinitialize the rest
            # extract the old sprites from g_theta
            old_sprites = trainer.model.sprites.masks_.proto.data
            old_proto = trainer.model.sprites.prototypes.data

            # init everything to rand
            trainer.model.selection.factoring.data = trainer.model.selection.init_factoring(factoring).to(trainer.model.selection.factoring.data.device)
            trainer.model.sprites.masks_.proto.data = torch.rand((len(alphabet), 1, *old_sprites.shape[-2:]), device=old_sprites.device)
            trainer.model.selection.n_sprites = len(alphabet)
        
            trainer.model.sprites.prototypes.data = trainer.model.sprites.init_proto(len(alphabet)).to(trainer.model.sprites.prototypes.data.device)
            trainer.model.sprites.prototypes.active_prototypes = torch.ones(len(alphabet))

            # copy the old sprites to the new ones
            for k, v in new_to_old_common_sprites.items():
                trainer.model.sprites.masks_.proto.data[k] = old_sprites[v]
                trainer.model.sprites.prototypes.data[k] = old_proto[v]

            # recompute the blank
            trainer.model.sprites.n_sprites = len(alphabet)
            trainer.model.blank = len(alphabet)
            if trainer.model.loss.ctc_factor > 0:
                from learnable_typewriter.typewriter.optim.loss import CTC
                trainer.model.loss.ctc = CTC(blank=trainer.model.blank)

            # save a json with the new index to uc match
            os.makedirs(save_dir, exist_ok=True)
            with open(join(save_dir, 'unique_combining.json'), 'w') as f:
                json.dump({v: k for k, v in unique_combining.items()}, f, indent=4)

            # do the same with the old index to alphabet match
            with open(join(save_dir, 'matching_old.json'), 'w') as f:
                json.dump(matching_old, f, indent=4)

            # TODO we need to handle saving so that it is performed in the correct way.
        else:
            raise

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

def eval_ft(trainer, path, split):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for images_to_tsf in trainer.get_dataloader(split=split, batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=False)[0]:
        decompose = trainer.decompositor
        obj = decompose(images_to_tsf)
        reco, seg = obj['reconstruction'].cpu(), obj['segmentation'].cpu()
        nrow = images_to_tsf['x'].size()[0]
        transformed_imgs = torch.cat([images_to_tsf['x'][:, :3].cpu().unsqueeze(0), reco.unsqueeze(0), seg.unsqueeze(0)], dim=0)
        transformed_imgs = torch.flatten(transformed_imgs, start_dim=0, end_dim=1)
        grid = make_grid(transformed_imgs, nrow=nrow)
        ToPILImage()(torch.clamp(grid, 0, 1)).save(path)
        return

def plot_sprites(trainer, drr, invert_sprites, save_individual=True, grid_pref='grid'):
    os.makedirs(drr, exist_ok=True)
    masks = trainer.model.sprites.masks
    if invert_sprites:
        masks = 1 - masks

    if save_individual:
        for k in range(len(trainer.model.sprites)):
            ToPILImage()(masks[k]).save(join(drr, f'{k}.png'))

    ToPILImage()(make_grid(masks, nrow=4)).save(join(drr, f'{grid_pref}.png'))
    ToPILImage()(make_grid(masks, nrow=len(trainer.model.sprites))).save(join(drr, f'{grid_pref}-1l.png'))

def stack(imgs):
    dst = PIL.Image.new('RGB', (imgs[0].width, len(imgs)*imgs[0].height))
    for i, img in enumerate(imgs):
        dst.paste(img, (0, i*imgs[0].height))
    return dst

def get_loader(trainer, split):
    return trainer.get_dataloader(split=split, batch_size=trainer.batch_size, num_workers=trainer.n_workers, shuffle=True)[0]

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
                    "name": args.opt,
                    "lr": 0,
                    "prototypes": {   
                        "lr": args.lr
                    },
                    "encoder": {'weight_decay': 0}
                }
            }
    elif args.mode == "g_theta" or args.mode == "g_theta+bkg":
        conf["training"] = {
            "optimizer": {
                "name": args.opt,
                "lr": args.lr,
                "finetune": args.mode
            }
        }

    conf["model"] = {
        "loss": {
            "ctc_factor": 0.0
        }
    }
    return conf

def make_factoring(alphabet): #creates a matrix with the alphabet and the combining characters
    mapping = {a: list(unicodedata.normalize('NFD', a)) for a in alphabet}
    unique_combining = list(sorted(set(v for vs in mapping.values() for v in vs)))
    return unique_combining

def log_transcribe(tag, values):
    print(tag.capitalize() + ':', json.dumps(list(values)))

def get_parser(name):
    parser = argparse.ArgumentParser(description=f'Helper script to finetune {name} on a specific set of documents or script and generate figures.')
    parser.add_argument('-i', "--model_path", type=str, required=True)
    parser.add_argument('-o', "--output_path", required=True, default=None)
    parser.add_argument('-r', "--reconstructions_path", default='reco', type=str)
    parser.add_argument('-s', "--sprites_path", default='sprites', type=str)
    parser.add_argument('-d', "--data_path", type=str, required=True)
    parser.add_argument('-a', "--annotation_file", required=True, default=join(LIB_PATH, 'datasets/south_north_textualis_mask/annotation.json'), type=str)
    parser.add_argument("--max_steps", required=False, default=2000, type=int)
    parser.add_argument("--log_every", required=False, default=500, type=int)
    parser.add_argument("--mode", choices=["all", "sprites", "g_theta", "g_theta+bkg"], default='g_theta')
    parser.add_argument("--invert_sprites", action='store_true')
    parser.add_argument("--enable_experimental", action='store_true')
    parser.add_argument("--device", default=0, type=str)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--split", choices=['train', 'val', 'all'], type=str, required=True)
    parser.add_argument("--opt", choices=['adam', 'sgd'], default='adam', type=str)
    
    return parser
