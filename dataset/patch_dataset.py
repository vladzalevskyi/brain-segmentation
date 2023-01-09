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
            raise NotImplementedError('Test dataset is not implemented yet')
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
                 normalization: str = 'z_score',
                 root_data_path: str = '/home/user0/misa_vlex/brain_segmentation/data',
                 use_all_data: bool = False):
        
        self.split = split
        self.use_all_data = use_all_data
        
        if split == 'train':
            self.img_dir = Path(f'{root_data_path}/Training_Set').resolve()

        # with training and validation set
        if self.use_all_data:
            self.img_dir = Path(f'{root_data_path}/FullTrainingSet').resolve()
        
        if split == 'val':
            self.img_dir = Path(f'{root_data_path}/Validation_Set').resolve()
        
        self.root_data_path = root_data_path
        self.window_size = window_size
        self.stride = stride
        self.denoiser = denoiser
        self.img_threshold = img_threshold
        self.normalization = normalization
        self.augmentation = augmentation
        self.load_images_patches()
        
        self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5,
                         border_mode=cv2.BORDER_CONSTANT,
                         value=0, interpolation=cv2.INTER_NEAREST),
                
                A.GaussianBlur(p=0.2,
                               blur_limit=(3,3),
                               sigma_limit=0.8),
                A.Downscale(p=0.3,
                              scale_min=0.8,
                              scale_max=0.9,
                              interpolation=cv2.INTER_NEAREST),
                A.RandomBrightnessContrast(p=0.5,
                                           brightness_limit=0.1,
                                           contrast_limit=0.1,
                                           brightness_by_max=True),
            ])
    
    def load_images_patches(self):
        self.img_patches = []
        self.mask_patches = []
        self.bbox_coords = []
        self.cases = []
        self.ssegm_patches = []
        
        for dir in self.img_dir.iterdir():
            if dir.is_dir() and 'unet' not in dir.name:
                case = dir.name
                # read image and mask
                img = sitk.GetArrayFromImage(sitk.ReadImage(str(dir / f'{case}.nii.gz')))
                segm = sitk.GetArrayFromImage(sitk.ReadImage(str(dir / f'{case}_seg.nii.gz')))
                
                if self.denoiser == 'synthseg' or self.denoiser == 'synthseg_merged':
                    ssegm_path = str(dir / f'{case}_seg.nii.gz')
                    repl_str = 'seg_resampled_merged' if self.denoiser == 'synthseg_merged' else 'seg_resampled'
                    ssegm_path = ssegm_path.replace('seg.nii.gz', f'{repl_str}.nii.gz')
                    ssegm_path = ssegm_path.replace('data', 'proc_data')
                    
                    if Path(str(ssegm_path).replace('FullTrainingSet', 'Training_Set')).exists():
                        ssegm_path = str(ssegm_path).replace('FullTrainingSet', 'Training_Set')
                    elif Path(str(ssegm_path).replace('FullTrainingSet', 'Validation_Set')).exists():
                        ssegm_path = str(ssegm_path).replace('FullTrainingSet', 'Validation_Set')
                    
                    ssgegm = sitk.GetArrayFromImage(sitk.ReadImage(ssegm_path))
                    _, ssegm_slices, __ = self.extract_patches(img, ssgegm)
                    self.ssegm_patches.extend(ssegm_slices)
                
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

    def augment(self, img, mask, idx):
        ssegmask = None
        if self.augmentation and self.split == 'train':
            img = utils.min_max_norm(img, 255).astype('uint8')
            # apply augmentations
            
            if self.denoiser == False:
                transformed = self.transform(image=img, mask=mask)
                
                mask = transformed['mask']
            elif self.denoiser == 'synthseg' or self.denoiser == 'synthseg_merged':
                transformed = self.transform(image=img,
                                             masks=[mask, self.ssegm_patches[idx]])
                
                mask, ssegmask = transformed['masks']
            else:
                raise ValueError(f'Denoiser type {self.denoiser} not implemented')
            img = transformed['image'].copy()

        else:
            img = img
            ssegmask = self.ssegm_patches[idx] if self.denoiser == 'synthseg' or self.denoiser == 'synthseg_merged'  else None
        
        return img, mask, ssegmask
        
    def __getitem__(self, idx):
        img = self.img_patches[idx]
        
        
        img, mask, ssegmask = self.augment(img, self.mask_patches[idx], idx)
            
        img = np.expand_dims(img, axis=0)
        
        
        if self.normalization == 'min_max':
            img = utils.min_max_norm(img, 1).astype('float32')
        elif self.normalization == 'z_score':
            img = utils.z_score_norm(img, non_zero_region=True)
        
        img = torch.Tensor(img)
        mask = torch.tensor(mask, dtype=torch.long)
        
        # add additional channel for synthseg segmentation
        if self.denoiser == 'synthseg' or self.denoiser == 'synthseg_merged' and ssegmask is not None:
            ssegmask = torch.tensor(ssegmask, dtype=torch.float)
            img = torch.cat([img, ssegmask.unsqueeze(0)], dim=0)
                
        return {'img': img,
                'mask': mask,
                'bbox': self.bbox_coords[idx],
                'case': self.cases[idx]}