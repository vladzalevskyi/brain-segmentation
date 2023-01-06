from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule, Trainer, LightningModule


from tqdm import tqdm
from dataset.patch_dataset import BrainPatchesDataModule
from models.UNetModule import UNet3
from dataset.roi_extraction import slice_image, reconstruct_patches
from utils import z_score_norm
import SimpleITK as sitk
import torch
from models.EM import ExpectationMaximization
import cv2
import matplotlib.pyplot as plt

cfg = {'pl_trainer':{'max_epochs': 20,
                     'devices': [0],
                     'accelerator': 'gpu'},
       
       'dataset':{'window_size': 64,
                  'stride': 32,
                  'img_threshold': 0.5,
                  'normalization': 'z_score'},
       
       'train_num_workers':8,
       'train_batch_size': 64,
       'val_num_workers':8,
       'val_batch_size': 64}

device = torch.device('cuda:2')
class BrainSegmenter:
    def __init__(self, model_checkpoint_path: Path,
                 cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.model = UNet3.load_from_checkpoint(model_checkpoint_path).to(device)

        # disable randomness, dropout, etc...
        self.model.eval()


    def segment(self, image: np.ndarray):
        segm_reconstructed = np.zeros_like(image)
        
        for s in tqdm(range(image.shape[0])):
                
            image_slices = slice_image(image[s, :, :],
                                    self.cfg['dataset']['window_size'],
                                    self.cfg['dataset']['stride'])
            
            empty_slices = [s.sum()==0 for s in image_slices]
            
            # CHANGE IF CHANGE NORMALIZATION OR ADD ANOTHER CHANNEL
            image_slices = [np.expand_dims(z_score_norm(slice), axis=0) for slice in image_slices]
            image_slices = torch.tensor(np.array(image_slices),
                                        requires_grad=False).to(device)
            # predict with the model
            y_hat = self.model(image_slices).detach().cpu().numpy()
            y_hat = np.argmax(y_hat, axis=1)
            # print(y_hat.shape, image_slices.shape, segm_reconstructed.shape)
            segm_reconstructed[s, :, :] = reconstruct_patches(y_hat,
                              image,
                              cfg['dataset']['window_size'],
                              cfg['dataset']['stride'])
            
        # print(y_hat.shape, image_slices.shape, segm_reconstructed.shape)
        return segm_reconstructed

if __name__ == '__main__':
    chk = '/home/user0/misa_vlex/brain_segmentation/outputs/lightning_logs/version_3/checkpoints/epoch=13-valid_dsc=0.9522.ckpt'
    brsm = BrainSegmenter(chk, cfg)
    img = sitk.ReadImage('/home/user0/misa_vlex/brain_segmentation/data/Validation_Set/IBSR_11/IBSR_11.nii.gz')
    segm = sitk.ReadImage('/home/user0/misa_vlex/brain_segmentation/data/Validation_Set/IBSR_11/IBSR_11_seg.nii.gz')
    
    em = ExpectationMaximization(3)
    segm_pred = brsm.segment(sitk.GetArrayFromImage(img))
    print(em.compute_dice(sitk.GetArrayFromImage(segm), segm_pred))
    plt.imshow(segm_pred[:, 80, :])
    plt.show()
    plt.imshow(sitk.GetArrayFromImage(segm)[:, 80, :])
    plt.show()
    