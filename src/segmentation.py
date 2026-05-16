"""
U-Net segmentation model for defect detection
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

from .config import seg_config, path_config
from .preprocessing import preprocess_image, extract_defect_features


class SegmentationModel:
    """U-Net model for defect segmentation"""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: Optional[Path]) -> nn.Module:
        """Load trained U-Net model"""
        model = smp.Unet(
            encoder_name=seg_config.encoder_name,
            encoder_weights=None,
            in_channels=seg_config.in_channels,
            classes=seg_config.num_classes,
            activation=None
        )
        
        # Nếu không chỉ định model_path, tìm file model
        if model_path is None:
            # Chuyển string thành Path object
            models_dir = Path(path_config.segmentation_models)
            
            # Thử các đường dẫn theo thứ tự ưu tiên
            candidate_paths = [
                models_dir / "best_model.pth",
                models_dir / "best_iou_model.pth",
                models_dir / "latest_model.pth"
            ]
            
            model_path = None
            for path in candidate_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"No segmentation model found in {models_dir}. "
                    f"Please train the model first using scripts/train_segmentation.py"
                )
        
        # Đảm bảo model_path là Path object
        model_path = Path(model_path)
        
        print(f"📦 Loading segmentation model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Lấy state_dict từ checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            # Nếu key bắt đầu bằng "model.", bỏ "model." đi
            if key.startswith('model.'):
                new_key = key[6:]  # Bỏ "model."
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # Load state_dict đã xử lý
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys[:5]}... (total: {len(missing_keys)})")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {unexpected_keys[:5]}... (total: {len(unexpected_keys)})")

        print(f"✅ Loaded segmentation model from {model_path}")
        model = model.to(self.device)
        return model
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to model input size
        input_h, input_w = seg_config.input_size
        resized = cv2.resize(gray, (input_w, input_h))
        
        # Normalize
        from .config import preprocess_config
        normalized = (resized.astype(np.float32) / 255.0 - preprocess_config.mean) / preprocess_config.std
        
        # Convert to tensor
        input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def predict(self, image: np.ndarray, mm_per_pixel: float = None) -> Dict:
        """
        Predict mask for a single image
        
        Returns:
            Dict: {
                'mask': np.ndarray (binary mask)
                'prob': np.ndarray (probability map)
                'num_defects': int
                'defects': list of defect info
            }
        """
        original_h, original_w = image.shape[:2]
        input_tensor = self._preprocess(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        
        # Resize back to original size
        mask = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask > seg_config.threshold).astype(np.uint8)
        
        # Extract defect features with original image size
        defect_features = extract_defect_features(
            binary_mask, 
            original_image_shape=(original_h, original_w),
            mm_per_pixel=mm_per_pixel
        )
        
        return {
            'mask': binary_mask,
            'prob': mask,
            'num_defects': defect_features['num_defects'],
            'defects': defect_features['defects'],
            'defect_features': defect_features
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict masks for a batch of images"""
        return [self.predict(img) for img in images]