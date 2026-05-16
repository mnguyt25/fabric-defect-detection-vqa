"""
Dataset classes for segmentation and VQA
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from .config import preprocess_config, seg_config


class DefectSegmentationDataset(Dataset):
    """Dataset for U-Net segmentation training"""
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.samples = []
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(img_file)[0]
                mask_file = f"{base_name}.png"
                
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)
                
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
        
        print(f"Found {len(self.samples)} pairs in {os.path.basename(image_dir)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = (mask > 0).long()
        
        return image, mask


def get_transforms(input_size=(512, 512)):
    """Get augmentation transforms for training"""
    
    train_transform = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], p=0.3),
        A.Normalize(mean=[preprocess_config.mean], std=[preprocess_config.std]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=[preprocess_config.mean], std=[preprocess_config.std]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def create_segmentation_dataloaders(
    data_root: str,
    train_dirs: list = None,
    val_dirs: list = None,
    batch_size: int = 8,
    input_size: tuple = (512, 512),
    train_ratio: float = 0.8
):
    """Create dataloaders for segmentation training"""
    
    image_root = os.path.join(data_root, 'segmentation')
    mask_root = os.path.join(data_root, 'mask')
    
    all_subdirs = [d for d in os.listdir(image_root) 
                   if os.path.isdir(os.path.join(image_root, d))]
    
    if train_dirs is None and val_dirs is None:
        random.seed(42)
        random.shuffle(all_subdirs)
        split_idx = int(len(all_subdirs) * train_ratio)
        train_dirs = all_subdirs[:split_idx]
        val_dirs = all_subdirs[split_idx:]
    
    train_transform, val_transform = get_transforms(input_size)
    
    # Create train datasets
    train_datasets = []
    for subdir in train_dirs:
        img_dir = os.path.join(image_root, subdir)
        msk_dir = os.path.join(mask_root, subdir)
        if os.path.exists(img_dir) and os.path.exists(msk_dir):
            ds = DefectSegmentationDataset(img_dir, msk_dir, train_transform)
            if len(ds) > 0:
                train_datasets.append(ds)
    
    # Create val datasets
    val_datasets = []
    for subdir in val_dirs:
        img_dir = os.path.join(image_root, subdir)
        msk_dir = os.path.join(mask_root, subdir)
        if os.path.exists(img_dir) and os.path.exists(msk_dir):
            ds = DefectSegmentationDataset(img_dir, msk_dir, val_transform)
            if len(ds) > 0:
                val_datasets.append(ds)
    
    if len(train_datasets) == 0:
        raise ValueError("No training data found!")
    if len(val_datasets) == 0:
        raise ValueError("No validation data found!")
    
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader