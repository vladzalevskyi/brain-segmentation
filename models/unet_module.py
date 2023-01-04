import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from models.UNet_3Plus import UNet_3Plus
import numpy as np
import torch
import torchmetrics

# define the LightningModule
class UNet3(pl.LightningModule):
    def __init__(self,
                 n_classes=4,
                 in_channels=1,
                 feature_scale=4):
        super().__init__()
        self.model = UNet_3Plus(in_channels=in_channels,
                                n_classes=n_classes,
                                feature_scale=feature_scale)

        self.criterion = nn.CrossEntropyLoss()
        
        self.train_dsc = torchmetrics.Dice(ignore_index=0,
                                           num_classes=n_classes,
                                           average='micro')
        
        self.valid_sc = torchmetrics.Dice(gnore_index=0,
                                          num_classes=n_classes,
                                          average='micro')
        
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['img']
        y = batch['mask']
        
        batch_size = len(y)
        y_hat = self.model(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        self.train_dsc(y_hat, y)
        self.log('train_dsc', self.train_dsc,
                 on_step=True,  on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


