import torch
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path

import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core import LightningDataModule
from dataset.roi_extraction import reconstruct_patches, extract_ROIs
import albumentations as A
import cv2 
logger = logging.getLogger(__name__)


class BrainPatchesDataModule(LightningDataModule):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.cfg = cfg
        if mode=='train':
            
            self.train_dataset = BrainPatchesDataset(split='train',
                                                     **cfg['dataset']['patches'])
            
            self.val_dataset = BrainPatchesDataset(split='val',
                                                     **cfg['dataset']['patches'])
            logger.info(
                f'Len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}'
            )
        else:
            # TODO: Define test dataset
            pass
            # self.test_dataset = self.DataSet(
            #     cfg.data_dir, split="test", chall=cfg.chall, cfg=cfg)
            # logger.info(f'len of test examples {len(self.test_dataset)}')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg['dataset']['train_batch_size'],
            shuffle=True,
            num_workers=self.cfg['dataset']['train_num_workers'])

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg['dataset']['val_batch_size'],
            shuffle=False,
            num_workers=self.cfg['dataset']['val_num_workers'])
        return val_loader


class BrainPatchesDataset(torch.utils.data.Dataset):
    def __init__(self, split: str,
                 window_size: int = 128,
                 stride: int = 64,
                 img_threshold: float = 0.1,
                 denoiser: bool = False,
                 augmentation: bool = False,
                 normalization: str = 'z_score'):
        
        if split == 'train':
            self.img_dir = Path('./data/Training_Set').resolve()
        if split == 'val':
            self.img_dir = Path('./data/Validation_Set').resolve()
        
        self.window_size = window_size
        self.stride = stride
        self.img_threshold = img_threshold
        self.normalization = normalization
        self.augmentation = augmentation
        self.load_images_patches()
        
        self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.2,
                         interpolation=cv2.INTER_NEAREST,
                         mask_value=0,
                         border_mode=cv2.BORDER_CONSTANT),
            ])
    
    def load_images_patches(self):
        self.img_patches = []
        self.mask_patches = []
        self.bbox_coords = []
        self.cases = []
        
        
        for dir in self.img_dir.iterdir():
            if dir.is_dir():
                case = dir.name
                # read image and mask
                img = sitk.GetArrayFromImage(sitk.ReadImage(str(dir / f'{case}.nii.gz')))
                segm = sitk.GetArrayFromImage(sitk.ReadImage(str(dir / f'{case}_seg.nii.gz')))
                
                # store slices per image-case
                img_slices, mask_slices, bbox_coords = self.extract_patches(img, segm)
                case = [dir.name]*len(img_slices)
                self.img_patches.extend(img_slices)
                self.mask_patches.extend(mask_slices)
                self.bbox_coords.extend(bbox_coords)
                self.cases.extend(case)
                
    def filter_rois(self, rois):
        """Removes patches that have less than img_threshold of the image filled"""
        img_prop = [(i>0).sum()/(i.shape[0]*i.shape[1]) for i in rois[0]]
        img_prop = np.asarray(img_prop)>self.img_threshold
        
        return [np.asarray(r)[img_prop] for r in rois]

    def extract_patches(self, img, segm):
        """Extracts patches from the image and mask
        and saves the coordinates of the bounding box
        of the patch in the original image."""
        img_slices = []
        mask_slices = []
        bbox_coords = []
        # Extract ROIs from the image X axis
        for s in range(img.shape[0]):
            rois = extract_ROIs(img[s, :, :],
                                segm[s, :, :],
                                window_size=self.window_size,
                                stride=self.stride)
            rois = self.filter_rois(rois)
            img_slices.extend(rois[0])
            mask_slices.extend(rois[1])
            bbox_coords.extend(rois[2])
            
        # Extract ROIs from the image Y axis
        for s in range(img.shape[1]):
            rois = extract_ROIs(img[:, s, :],
                                segm[:, s, :],
                                window_size=self.window_size,
                                stride=self.stride)
            rois = self.filter_rois(rois)
            img_slices.extend(rois[0])
            mask_slices.extend(rois[1])
            bbox_coords.extend(rois[2])
            
        # Extract ROIs from the image Z axis
        for s in range(img.shape[2]):
            rois = extract_ROIs(img[:, :, s],
                                segm[:, :, s],
                                window_size=self.window_size,
                                stride=self.stride)
            rois = self.filter_rois(rois)
            img_slices.extend(rois[0])
            mask_slices.extend(rois[1])
            bbox_coords.extend(rois[2])
        return img_slices, mask_slices, bbox_coords
            
    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, idx):
        img = self.img_patches[idx]
        
        # FOR NOW JUST EXPAND DIMS
        # LATER COULD ADD ROUGH SEGM
        img = np.expand_dims(img, axis=0)
        if self.normalization == 'min_max':
            img = utils.min_max_norm(img, 1).astype('float32')
        elif self.normalization == 'z_score':
            img = utils.z_score_norm(img, non_zero_region=True)
        
        if self.augmentation:
            # apply augmentations
            transformed = self.transform(image=img,
                                        mask=self.mask_patches[idx])
            img = transformed['image'].copy()
            mask = transformed['mask']
        else:
            img = img#transformed['image'].copy()
            mask = self.mask_patches[idx]#transformed['mask']
        return {'img': torch.Tensor(img),
                'mask': torch.tensor(mask, dtype=torch.long),
                'bbox': self.bbox_coords[idx],
                'case': self.cases[idx]}