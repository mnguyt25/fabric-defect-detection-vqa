#!/usr/bin/env python
"""
Script kiểm tra hệ thống phát hiện lỗi
"""
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation import SegmentationModel
from src.config import path_config

def check_system():
    print("="*60)
    print("🔍 KIỂM TRA HỆ THỐNG")
    print("="*60)
    
    # 1. Kiểm tra model segmentation
    models_dir = Path(path_config.segmentation_models)  # ✅ Chuyển thành Path
    print(f"\n📁 Thư mục models: {models_dir}")
    
    model_files = ['best_model.pth', 'best_iou_model.pth', 'latest_model.pth']
    found = False
    
    for f in model_files:
        model_path = models_dir / f
        if model_path.exists():
            size = model_path.stat().st_size / (1024*1024)
            print(f"   ✅ Found {f} ({size:.1f} MB)")
            found = True
        else:
            print(f"   ❌ Missing {f}")
    
    if not found:
        print("\n⚠️ CHƯA CÓ MODEL SEGMENTATION!")
        print("   Vui lòng chạy: python scripts/02_train_segmentation.py")
        return False
    
    # 2. Thử load model
    print("\n🚀 Thử load model segmentation...")
    try:
        model = SegmentationModel(device="cpu")
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return False
    
    print("\n✅ Hệ thống sẵn sàng!")
    return True

if __name__ == "__main__":
    check_system()