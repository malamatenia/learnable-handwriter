import os
import sys
import json
import PIL
import torch

from tqdm import tqdm
from os.path import dirname, abspath, join, isfile
from collections import defaultdict
from finetune_utils import check_patch, plot_sprites, make_optimizer_conf, get_parser
from scripts.finetune_scripts import finetune

LIB_PATH = join(dirname(dirname(abspath(__file__))))
sys.path.append(LIB_PATH)
from learnable_typewriter.utils.loading import load_pretrained_model

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

def run(args):
    documents = get_documents(args.annotation_file, args.split, args.filter)
    print(f"Filtered Documents size: {len(documents)}")
    os.makedirs(join(args.output_path, args.sprites_path), exist_ok=True)
    pbar = tqdm(documents)
    
    for document, split in documents:
        pbar.set_description(f"Processing {document}")

        trainer = load_pretrained_model(path=args.model_path, device=str(args.device), conf=make_optimizer_conf(args))

        from omegaconf import OmegaConf
        config_file = join(args.output_path, 'config.yaml')
        with open(config_file, 'w') as f:
            f.write(OmegaConf.to_yaml(trainer.cfg))

        os.makedirs(join(args.output_path, args.sprites_path, 'transcribe'), exist_ok=True)
        transcribe_file = join(args.output_path, args.sprites_path, 'transcribe', f'{document}.json')
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

        # Update save_model_dir to save 'model.pth' in the document's directory
        finetune(trainer, max_steps=args.max_steps, log_every=args.log_every, 
                 save_sprites_dir=join(args.output_path, document, args.sprites_path), 
                 reconstructions_path=join(args.output_path, document, args.reconstructions_path), 
                 save_model_dir=join(args.output_path, document, 'model.pth'), 
                 invert_sprites=args.invert_sprites, split=split)
                 
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
    parser = get_parser('docs')
    parser.add_argument('-f', "--filter", nargs="+", default=None)
    args = parser.parse_args()

    run(args)
