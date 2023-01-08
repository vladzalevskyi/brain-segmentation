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

from scipy.stats import mode

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
                progress: bool = False,
                ssegm_image=None) -> np.ndarray:
        
        segm_reconstructed = np.zeros_like(image)
        
        if progress:
            range_iter = tqdm(range(image.shape[0]))
        else:
            range_iter = range(image.shape[0])
        for slice in range_iter:
            
            image_slices = slice_image(image[slice, :, :],
                                       self.cfg['dataset']['patches']['window_size'],
                                       self.cfg['dataset']['patches']['stride'])
            
            image_slices = [z_score_norm(slice, non_zero_region=True) for slice in image_slices]
            image_slices = np.expand_dims(np.asarray(image_slices, dtype=np.float32), axis=1)
            image_slices = torch.tensor(image_slices, requires_grad=False).to(self.device)
            
            if self.cfg['model']['in_channels'] == 2 and ssegm_image is not None:
                ssegm_slices = slice_image(ssegm_image[slice, :, :],
                                           self.cfg['dataset']['patches']['window_size'],
                                           self.cfg['dataset']['patches']['stride'])
                ssegm_slices = torch.tensor(ssegm_slices, dtype=torch.float).to(self.device)
                ssegm_slices = ssegm_slices.unsqueeze(1)
                image_slices = torch.cat((image_slices, ssegm_slices), dim=1)
            
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
                            mask: np.ndarray,
                            ssegm_image: np.ndarray|None = None,
                            ensemble: bool=False):
        if ensemble:
            mask_pred = self.segment_ensemble(image, progress=False, ssegm_image_xyz=ssegm_image)
        else:
            mask_pred = self.segment(image, progress=False, ssegm_image=ssegm_image)
        
        res = ExpectationMaximization.compute_dice(mask, mask_pred,
                                                   map_dict={1:1, 2:2, 3:3})
        
        res['avg_dice'] = np.mean(list(res.values()))
        
        return mask_pred, res
    
    def segment_ensemble(self, img_xyz: np.ndarray,
                         progress: bool = False,
                         ssegm_image_xyz=None) -> np.ndarray:
        
        # rearrange axes with numpy
        img_yzx = np.transpose(img_xyz, (1, 2, 0))
        img_zxy = np.transpose(img_xyz, (2, 0, 1))
        
        if ssegm_image_xyz is not None:
            ssegm_yzx = np.transpose(ssegm_image_xyz, (1, 2, 0))
            ssegm_zxy = np.transpose(ssegm_image_xyz, (2, 0, 1))
        else:
            ssegm_yzx = None
            ssegm_zxy = None
        
        pred_xyz = self.segment(img_xyz,
                                progress=progress,
                                ssegm_image=ssegm_image_xyz)
        pred_yzx = self.segment(img_yzx,
                                progress=progress,
                                ssegm_image=ssegm_yzx)
        pred_zxy = self.segment(img_zxy,
                                progress=progress,
                                ssegm_image=ssegm_zxy)
        
        pred_xyz2 = np.transpose(pred_yzx, (2, 0, 1))
        pred_xyz3 = np.transpose(pred_zxy, (1, 2, 0))
        
        ens_pred = np.stack((pred_xyz, pred_xyz2, pred_xyz3), axis=0)
        final_pred, _ = mode(ens_pred, axis=0, keepdims=False)
        return final_pred