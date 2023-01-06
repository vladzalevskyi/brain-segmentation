
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from IPython.core.display import display
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import kornia.augmentation as K

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = K.AugmentationSequential(
            K.RandomRotation(degrees=90.0, p=0.5),
            K.RandomThinPlateSpline(p=0.5),
            data_keys=["input", "mask"],  # Just to define the future input here.

            )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.transforms(x, y.float())
        return out