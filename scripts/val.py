# Different from a eval this is used to run a single validation step from a pretrained model.
from os.path import join, dirname, abspath
import sys; sys.path.append(join(dirname(dirname(abspath(__file__)))))
from learnable_typewriter.utils.loading import load_pretrained_model

def run(args):
    trainer = load_pretrained_model(path=args.input_path, device=str(args.device), kargs=args.kargs)
    trainer.evaluate_only = True
    trainer.train_end = True
    trainer.replace_run_dir_(args.output_path)
    trainer.__init_metrics__()
    trainer.__init_tensorboard__(force=True)
    trainer.log_train_metrics()
    trainer.log_val_metrics()
    trainer.log_images()
    trainer.error_rate()
    trainer.__close_tensorboard__()


if __name__ == "__main__":
    import argparse, torch

    parser = argparse.ArgumentParser(description='Generic tool for quantitative (reco-error, error-rate) and qualitative (reconstruction, segmentation, sprites) evaluation.')
    parser.add_argument('-i', "--input_path", required=True, default=None, help='Model Path')
    parser.add_argument('-d', "--device", default=(0 if torch.cuda.is_available() else 'cpu'), type=str, help='Cuda ID')
    parser.add_argument('-o', "--output_path", required=False, default='vals', type=str, help='Save Output dir')
    parser.add_argument("--kargs", type=str, default='training.batch_size=16', help='Override the loaded kwargs with an injected OmegaConf profile.')
    args = parser.parse_args()

    run(args)
