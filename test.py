import os
import argparse
import torch
import pytorch_lightning as pl

from model import UNet
from dataset import get_loader

def train(args):
    use_cuda = ('cuda' in args.device) and torch.cuda.is_available() and (int(args.device[5:]) < torch.cuda.device_count() if len(args.device) > 5 else True)
    pl.seed_everything(args.seed, workers = True)
    trainer = pl.Trainer(default_root_dir = os.path.join(args.log, args.experiment),
                         benchmark = True,
                         accelerator = 'gpu' if use_cuda else 'cpu',
                         devices = torch.cuda.device_count() if use_cuda and len(args.device) == 4 else [int(args.device[5:])],
                         enable_progress_bar = True,
                         log_every_n_steps = 1,
                         )

    dataloader = get_loader(args, 'test', use_cuda)

    args.num_training_steps = 1
    model = UNet(args)

    trainer.test(model, dataloader,
                 ckpt_path = args.checkpoint if os.path.exists(args.checkpoint) else None)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model related
    parser.add_argument('--encoder', type = str, default = 'resnet50',
                        choices = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'mobilenet_v2', 'vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help = 'Encoder to be used in UNet')
    parser.add_argument('--weights', type = str, default = 'imagenet')
    parser.add_argument('--classes', type = int, default = 14)
    parser.add_argument('--activation', type = str, default = None)
    parser.add_argument('--in_channels', type = int, default = 28)

    # Hyperparameters
    parser.add_argument('--lr', type = float, default = 1e-3,
                        help = 'Learning rate')
    parser.add_argument('--warmup', type = int, default = 100,
                        help = 'No. of warmup steps in cosine scheduler')
    
    # Training related
    parser.add_argument('--experiment', type = str, required = True,
                        help = 'Experiment name for tracking')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoints',
                        help = 'Path where checkpoints are to be saved')
    parser.add_argument('--log', type = str, default = 'logs',
                        help = 'Path where logs are to be saved')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'Seed value for reproducibility of experiments')
    parser.add_argument('--device', type = str, default = 'cuda:0', choices = ['cuda:0', 'cuda', 'cpu'],
                        help = 'Device to use for training')
    parser.add_argument('--workers', type = int, default = 12,
                        help = 'No. of worker processes for data loading')
    
    # Data related
    parser.add_argument('--root', type = str, required = True,
                        help = 'Root directory of data')
    parser.add_argument('--json_list', type = str, default = 'ours2.json',
                        help = 'Dataset json file')
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'Input batch size')
    parser.add_argument('--a_min', type = float, default = -175.0,
                        help = 'a_min in ScaleIntesityRange')
    parser.add_argument('--a_max', type = float, default = -250.0,
                        help = 'a_max in ScaleIntesityRange')
    parser.add_argument('--b_min', type = float, default = 0.0,
                        help = 'b_min in ScaleIntesityRange')
    parser.add_argument('--b_max', type = float, default = 1.0,
                        help = 'b_max in ScaleIntesityRange')
    
    args = parser.parse_args()

    train(args)

