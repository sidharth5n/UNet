import torch
import segmentation_models_pytorch as smp
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import torchmetrics

class UNet(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr
        self.num_classes = args.num_classes
        self.warmup = args.warmup
        self.num_training_steps = args.num_training_steps
        self.model = smp.Unet(encoder_name = args.encoder,
                              encoder_weights = args.weights,
                              decoder_use_batchnorm = True,
                              classes = args.classes,
                              activation = args.activation,
                              in_channels = args.in_channels)
        self.loss_fn = smp.losses.DiceLoss(mode = 'multiclass')
        self.dice_fn = torchmetrics.Dice(num_classes = args.classes)
        self.accuracy_fn = torchmetrics.Accuracy(task = 'multiclass',
                                                 num_classes = args.classes)
    
    def training_step(self, batch, batch_idx):
        img, label = batch['image'], batch['label'].long()
        output = self.model(img)
        loss = self.loss_fn(output, label)
        self.log('train/loss', loss, on_step = True, on_epoch = False, prog_bar = True, logger = True)
        self.log('train/lr', self.lr_schedulers().get_last_lr()[0], on_step = True, on_epoch = False, prog_bar = True, logger = True)
    
    def validation_step(self, batch, batch_idx):
        img, label = batch['image'], batch['label'].long()
        with torch.no_grad():
            output = self.model(img)
        loss = self.loss_fn(output, label)
        dice = self.dice_fn(output, label)
        accuracy = self.accuracy_fn(output.argmax(dim = 1, keepdim = True), label)
        self.log('val/loss', loss, on_step = False, on_epoch = True, prog_bar = True, logger = True, reduce_fx = torch.sum)
        self.log('val/dice', dice, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log('val/dice', accuracy, on_step = False, on_epoch = True, prog_bar = True, logger = True)

    def test_step(self, batch, batch_idx):
        img, label = batch['image'], batch['label'].long()
        with torch.no_grad():
            output = self.model(img)
        loss = self.loss_fn(output, label)
        dice = self.dice_fn(output, label)
        accuracy = self.accuracy_fn(output.argmax(dim = 1, keepdim = True), label)
        self.log('test/loss', loss, on_step = False, on_epoch = True, prog_bar = True, logger = True, reduce_fx = torch.sum)
        self.log('test/dice', dice, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        self.log('test/dice', accuracy, on_step = False, on_epoch = True, prog_bar = True, logger = True)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()),
                                     lr = self.lr,
                                     weight_decay = 0)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = self.warmup,
                                                    num_training_steps = self.num_training_steps)
        
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler}