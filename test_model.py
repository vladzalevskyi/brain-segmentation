import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from models.UNet_3Plus import UNet_3Plus
import numpy as np
import torch

# define the LightningModule
class UNet3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet_3Plus(in_channels=2,
                                n_classes=4,
                                feature_scale=4)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # init module
    module = UNet3()
    a = torch.tensor(np.ones((2, 128, 128), dtype=np.float32))
    print(a.shape)
    a = a.unsqueeze(0)
    print(a.shape)
    print(module.model((a)).shape)
    
    # # init a trainer
    # trainer = pl.Trainer(gpus=1, max_epochs=3)

    # # train
    # trainer.fit(module)