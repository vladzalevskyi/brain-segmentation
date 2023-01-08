import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from models.UNet_3Plus import UNet_3Plus
from models.UNet_2Plus import UNet_2Plus
from models.UNet import UNet
import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt
import kornia.losses as losses
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

def get_criterions(name: str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'dice':
        return losses.DiceLoss()
    elif name == 'focal':
        return losses.FocalLoss(0.5)
    elif name == 'tversky':
        return losses.TverskyLoss(0.4, 0.4)
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_model(name: str):
    if name == 'UNet_3Plus':
        return UNet_3Plus
    elif name == 'UNet_2Plus':
        return UNet_2Plus
    elif name == 'UNet':
        return UNet
    else:
        raise ValueError(f'Unknown model name: {name}')

# define the LightningModule
class UNet3(pl.LightningModule):
    def __init__(self,
                 n_classes:int,
                 in_channels:int,
                 feature_scale:int,
                 loss: str,
                 model: str,
                 lr:float = 1e-3,):
        super().__init__()
        
        
        self.model = get_model(model)(in_channels=in_channels,
                                n_classes=n_classes,
                                feature_scale=feature_scale)
        
        self.criterion = get_criterions(loss)
        
        self.train_dsc_macro = torchmetrics.Dice(ignore_index=0,
                                                 num_classes=n_classes,
                                                 average='macro')
        
        
        self.valid_dsc_macro = torchmetrics.Dice(ignore_index=0,
                                                 num_classes=n_classes,
                                                 average='macro')
        self.valid_dsc_weighted= torchmetrics.Dice(ignore_index=0,
                                                 num_classes=n_classes,
                                                 average='weighted')
        
        self.learning_rate = lr
        
        self.save_hyperparameters()
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['img']
        y = batch['mask']
        
        batch_size = len(y)
        y_hat = self.model(x)
        
    
        loss = self.criterion(y_hat, y).mean()
        
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        self.train_dsc_macro(y_hat, y)
        self.log('train_dsc_macro', self.train_dsc_macro,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['img']
        y = batch['mask']
        
        batch_size = len(y)
        y_hat = self.model(x)
        
        loss = self.criterion(y_hat, y).mean()
        
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        self.valid_dsc_macro(y_hat, y)
        self.log('valid_dsc_macro', self.valid_dsc_macro,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        
        self.valid_dsc_weighted(y_hat, y)
        self.log('valid_dsc_weighted', self.valid_dsc_weighted,
                 on_step=False,  on_epoch=True,
                 prog_bar=False, logger=True,
                 batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.1, patience=10)
         #learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch,
                                 "monitor":"val_loss"}}

