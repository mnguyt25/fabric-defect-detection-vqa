"""
Model definitions for segmentation and VQA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import BertModel


class SegmentationModel(nn.Module):
    """U-Net model for defect segmentation"""
    
    def __init__(self, encoder_name='efficientnet-b0', num_classes=2, in_channels=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_encoder_features(self, x):
        """Extract encoder features for VQA"""
        features = self.model.encoder(x)
        return features
