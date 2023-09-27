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
                         max_epochs = args.epochs,
                         enable_checkpointing = True,
                         enable_progress_bar = True,
                         val_check_interval = args.validation_interval,
                         inference_mode = False,
                         log_every_n_steps = 1,
                         callbacks = [pl.callbacks.ModelCheckpoint(dirpath = os.path.join(args.checkpoint, args.experiment),
                                                                   monitor = 'val/loss',
                                                                   filename = 'model_best_val_loss'),
                                      pl.callbacks.ModelCheckpoint(dirpath = os.path.join(args.checkpoint, args.experiment),
                                                                   monitor = 'val/dice',
                                                                   filename = 'model_best_val_dice'),
                                      pl.callbacks.ModelCheckpoint(dirpath = os.path.join(args.checkpoint, args.experiment),
                                                                   filename = 'model_{epoch:02d}',
                                                                   every_n_epochs = 1),
                                      pl.callbacks.ModelCheckpoint(dirpath = os.path.join(args.checkpoint, args.experiment),
                                                                   monitor = 'train/loss',
                                                                   filename = 'model_best_train'),])

    train_dataloader = get_loader(args, 'train', use_cuda)
    val_dataloader = get_loader(args, 'val', use_cuda)

    args.num_training_steps = args.epochs * len(train_dataloader)
    model = UNet(args)

    ckpt_path = os.path.join(args.checkpoint, args.experiment, args.checkpoint_name)
    trainer.fit(model, train_dataloader, val_dataloader,
                ckpt_path = ckpt_path if os.path.exists(ckpt_path) and args.resume else None)
    
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
    parser.add_argument('--checkpoint_interval', type = int, default = 20,
                        help = 'Frequency (in steps) of saving checkpoints')
    parser.add_argument('--resume', default = False, action = 'store_true',
                        help = 'Whether to resume training from provided checkpoint')
    parser.add_argument('--checkpoint_name', type = str, default = '',
                        help = 'Checkpoint from whihch training is to be resumed')
    parser.add_argument('--log', type = str, default = 'logs',
                        help = 'Path where logs are to be saved')
    parser.add_argument('--log_interval', type = int, default = 1,
                        help = 'Frequency (in steps) of logging information')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'Seed value for reproducibility of experiments')
    parser.add_argument('--device', type = str, default = 'cuda:0', choices = ['cuda:0', 'cuda', 'cpu'],
                        help = 'Device to use for training')
    parser.add_argument('--workers', type = int, default = 12,
                        help = 'No. of worker processes for data loading')
    parser.add_argument('--epochs', type = int, default = 25,
                        help = 'No. of epochs to train')
    parser.add_argument('--validation_interval', type = int, default = 500,
                        help = 'Frequency (in steps) of performing validation')
    
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
    parser.add_argument('--roi_x', type = int, default = 96,
                        help = 'roi size in x direction')
    parser.add_argument('--roi_y', type = int, default = 96,
                        help = 'roi size in y direction')
    parser.add_argument('randflip_prob', default = 0.2, type = float, 
                        help = 'RandFlip augmentation probability')
    parser.add_argument('randrotate90_prob', default = 0.2, type = float, 
                        help = 'RandRotate90 augmentation probability')
    parser.add_argument('randscaleintensity_prob', default = 0.2, type = float, 
                        help = 'RandScaleIntensity augmentation probability')
    parser.add_argument('randshiftintensity_prob', default = 0.2, type = float, 
                        help = 'RandShiftIntensity augmentation probability')
    
    args = parser.parse_args()

    train(args)

