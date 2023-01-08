from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule, Trainer, LightningModule


from tqdm import tqdm
from dataset.patch_dataset import BrainPatchesDataModule
from models.UNetModule import get_model, UNet3
from dataset.roi_extraction import slice_image, reconstruct_patches
from utils import z_score_norm
import SimpleITK as sitk
import torch
from models.EM import ExpectationMaximization
import cv2
import matplotlib.pyplot as plt
import yaml
import time

class BrainSegmenter:
    def __init__(self, model_checkpoint_path: Path,
                 device: str = 'cuda:2'):
        """Constructor for BrainSegmenter class

        Args:
            model_checkpoint_path (Path): Path to lightning model checkpoint.
                config_dump.yaml file should be in the 
                /outputs/experiment_name folder 
        """
        super().__init__()
        self.device = device
        
        with open(model_checkpoint_path.parent.parent.parent/'config_dump.yml', 'r') as f:
            self.cfg  = list(yaml.load_all(f, yaml.SafeLoader))[0]

        self.model = UNet3(**self.cfg['model'])
        self.model = self.model.load_from_checkpoint(model_checkpoint_path,
                                                     **self.cfg['model']).to(self.device)

        # disable randomness, dropout, etc...
        self.model.eval()
        print('Model loaded')

    def segment(self, image: np.ndarray,
                progress: bool = False) -> np.ndarray:
        
        segm_reconstructed = np.zeros_like(image)
        
        if progress:
            range_iter = tqdm(range(image.shape[0]))
        else:
            range_iter = range(image.shape[0])
        for slice in range_iter:
            
            image_slices = slice_image(image[slice, :, :],
                                       self.cfg['dataset']['patches']['window_size'],
                                       self.cfg['dataset']['patches']['stride'])
                        
            # CHANGE IF CHANGE NORMALIZATION OR ADD ANOTHER CHANNEL
            image_slices = [z_score_norm(slice, non_zero_region=True) for slice in image_slices]
            image_slices = np.expand_dims(np.asarray(image_slices, dtype=np.float32), axis=1)
            image_slices = torch.tensor(image_slices, requires_grad=False).to(self.device)
            
            # predict with the model
            y_hat = self.model(image_slices).detach().cpu().numpy()
            y_hat = np.argmax(y_hat, axis=1)
            
            segm_reconstructed[slice, :, :] = reconstruct_patches(y_hat,
                              image[slice, :, :],
                              self.cfg['dataset']['patches']['window_size'],
                              self.cfg['dataset']['patches']['stride'])
        segm_reconstructed[image == 0] = 0
        return segm_reconstructed
    
    def segment_and_compare(self,
                            image: np.ndarray,
                            mask: np.ndarray):

        mask_pred = self.segment(image, progress=False)
        res = ExpectationMaximization.compute_dice(mask, mask_pred,
                                                   map_dict={1:1, 2:2, 3:3})
        res['avg_dice'] = np.mean(list(res.values()))
        return mask_pred, res