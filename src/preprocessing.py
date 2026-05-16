"""
Image preprocessing functions for defect detection
"""
import cv2
import numpy as np
import json
import os
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import pandas as pd

from .config import preprocess_config, path_config


def resize_with_pad(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 0,
    is_mask: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Resize image while preserving aspect ratio with padding
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    if len(image.shape) == 2:
        padded = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    else:
        padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=np.uint8)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    if len(image.shape) == 2:
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
    
    metadata = {
        'scale': scale,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'orig_size': (w, h),
        'new_size': (new_w, new_h),
        'target_size': target_size
    }
    
    return padded, metadata


def preprocess_image(
    image: np.ndarray,
    for_segmentation: bool = True,
    use_padding: bool = True
) -> Dict[str, np.ndarray]:
    """
    Preprocess a single image for feature extraction or segmentation
    """
    cfg = preprocess_config
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Median filter
    if cfg.median_kernel > 1:
        kernel = cfg.median_kernel if cfg.median_kernel % 2 == 1 else cfg.median_kernel + 1
        gray = cv2.medianBlur(gray, kernel)
    
    # CLAHE contrast enhancement
    if cfg.clahe:
        clahe_obj = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit, 
            tileGridSize=cfg.clahe_grid_size
        )
        gray = clahe_obj.apply(gray)
    
    # Resize
    if for_segmentation:
        target_size = cfg.resize_for_segmentation
    else:
        target_size = cfg.resize_for_feature
    
    if use_padding:
        resized, metadata = resize_with_pad(gray, target_size, is_mask=False)
    else:
        resized = cv2.resize(gray, target_size)
        metadata = None
    
    # Normalize
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - cfg.mean) / cfg.std
    
    result = {
        'image': resized,
        'metadata': metadata
    }
    
    if for_segmentation:
        result['target_size'] = cfg.resize_for_segmentation
    else:
        result['target_size'] = cfg.resize_for_feature
    
    return result


def preprocess_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
    use_padding: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess mask for segmentation training
    """
    if use_padding:
        resized, metadata = resize_with_pad(mask, target_size, pad_value=0, is_mask=True)
    else:
        resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        metadata = None
    
    return resized, metadata


def calculate_real_size(
    area_pixels: int,
    bbox_pixels: Tuple[int, int, int, int],
    image_size_pixels: Tuple[int, int],
    reference_mm_per_pixel: float = None
) -> Dict[str, float]:
    """
    Tính kích thước thực tế từ pixel sang mm
    
    Args:
        area_pixels: Diện tích trong ảnh (pixel^2)
        bbox_pixels: Bounding box (x, y, w, h) trong pixel
        image_size_pixels: Kích thước ảnh (width, height) pixel
        reference_mm_per_pixel: Tỷ lệ mm/pixel (nếu None thì dùng giả định)
    
    Returns:
        Dict với các giá trị kích thước thực
    """
    if reference_mm_per_pixel is None:
        mm_per_pixel = 0.1  # Default: 1 pixel = 0.1mm
    else:
        mm_per_pixel = reference_mm_per_pixel
    
    x, y, w, h = bbox_pixels
    
    return {
        'width_mm': w * mm_per_pixel,
        'height_mm': h * mm_per_pixel,
        'area_mm2': area_pixels * mm_per_pixel * mm_per_pixel,
        'mm_per_pixel': mm_per_pixel
    }


def extract_defect_features(
    mask: np.ndarray, 
    original_image_shape: Tuple[int, int] = None,
    mm_per_pixel: float = None
) -> Dict:
    """
    Extract geometric features from segmentation mask with real-world measurements
    """
    if mask.dtype == np.float32:
        mask = (mask > 0.5).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    num_defects = num_labels - 1
    
    if num_defects == 0:
        return {
            'num_defects': 0,
            'defects': [],
            'total_area': 0,
            'max_area': 0,
            'min_area': 0,
            'avg_area': 0,
            'total_area_mm2': 0,
            'severity': 'none'
        }
    
    # Kích thước ảnh gốc
    if original_image_shape is None:
        img_h, img_w = mask.shape
    else:
        img_h, img_w = original_image_shape[0], original_image_shape[1]
    
    defects = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])
        
        # Tính kích thước thực
        real_size = calculate_real_size(
            area_pixels=area,
            bbox_pixels=(x, y, w, h),
            image_size_pixels=(img_w, img_h),
            reference_mm_per_pixel=mm_per_pixel
        )
        
        # Mô tả vị trí
        position_desc = get_position_description((cx, cy), (img_w, img_h))
        
        defects.append({
            'id': i - 1,
            'area': area,
            'area_mm2': real_size['area_mm2'],
            'width_mm': real_size['width_mm'],
            'height_mm': real_size['height_mm'],
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'position': position_desc,
            'horizontal_position': 'left' if cx < img_w * 0.25 else ('right' if cx > img_w * 0.75 else 'center'),
            'vertical_position': 'top' if cy < img_h * 0.25 else ('bottom' if cy > img_h * 0.75 else 'center')
        })
    
    defects.sort(key=lambda d: d['area'], reverse=True)
    areas = [d['area'] for d in defects]
    areas_mm2 = [d['area_mm2'] for d in defects]
    
    # Đánh giá mức độ nghiêm trọng
    total_area_mm2 = sum(areas_mm2)
    if total_area_mm2 > 100:  # >100 mm²
        severity = 'critical'
    elif total_area_mm2 > 50:  # 50-100 mm²
        severity = 'high'
    elif total_area_mm2 > 10:  # 10-50 mm²
        severity = 'medium'
    elif num_defects > 0:
        severity = 'low'
    else:
        severity = 'none'
    
    return {
        'num_defects': num_defects,
        'defects': defects,
        'total_area': sum(areas),
        'total_area_mm2': total_area_mm2,
        'max_area': max(areas) if areas else 0,
        'min_area': min(areas) if areas else 0,
        'avg_area': sum(areas) / num_defects if num_defects > 0 else 0,
        'severity': severity
    }


def get_position_description(centroid: Tuple[float, float], image_size: Tuple[int, int]) -> str:
    """
    Describe defect position based on centroid coordinates (in English)
    
    Args:
        centroid: (x, y) coordinates of defect center
        image_size: (width, height) of image
    
    Returns:
        English position description (e.g., "top-left corner", "center", etc.)
    """
    x, y = centroid
    w, h = image_size
    
    # Horizontal position
    if x < w * 0.25:
        horizontal = "left"
        horizontal_detail = "left side"
    elif x > w * 0.75:
        horizontal = "right"
        horizontal_detail = "right side"
    else:
        horizontal = "center"
        horizontal_detail = "center"
    
    # Vertical position
    if y < h * 0.25:
        vertical = "top"
        vertical_detail = "top"
    elif y > h * 0.75:
        vertical = "bottom"
        vertical_detail = "bottom"
    else:
        vertical = "center"
        vertical_detail = "center"
    
    # Combine based on position
    if vertical == "center" and horizontal == "center":
        return "center of the fabric"
    elif vertical == "center":
        return f"{horizontal_detail} of the fabric"
    elif horizontal == "center":
        return f"{vertical_detail} of the fabric"
    elif vertical == "top" and horizontal == "left":
        return "top-left corner"
    elif vertical == "top" and horizontal == "right":
        return "top-right corner"
    elif vertical == "bottom" and horizontal == "left":
        return "bottom-left corner"
    elif vertical == "bottom" and horizontal == "right":
        return "bottom-right corner"
    else:
        return f"{vertical}-{horizontal} area"