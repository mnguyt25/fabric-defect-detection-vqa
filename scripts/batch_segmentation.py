#!/usr/bin/env python
"""
Script 12: Chạy segmentation trên toàn bộ ảnh sau khi fine-tune
Hỗ trợ lưu metadata theo cấu trúc thư mục con
"""
import os
import sys
import cv2
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import path_config, seg_config
from src.models import SegmentationModel
from src.preprocessing import extract_defect_features

torch.serialization.add_safe_globals([np.dtype, np.float32, np.int64, np.void])

class BatchSegmentationRunner:
    """
    Chạy segmentation trên toàn bộ ảnh sau khi fine-tune
    Hỗ trợ cấu trúc thư mục con để lưu metadata
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cpu',
                 output_dir: str = None,
                 preserve_subdirs: bool = True):
        """
        Khởi tạo batch segmentation runner
        
        Args:
            model_path: Đường dẫn đến file model .pth đã fine-tune
            device: 'cuda' hoặc 'cpu'
            output_dir: Thư mục lưu kết quả (mặc định: ./data/inference_results)
            preserve_subdirs: Giữ nguyên cấu trúc thư mục con khi lưu kết quả
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.preserve_subdirs = preserve_subdirs
        print(f"Using device: {self.device}")
        print(f"Preserve subdirectories: {self.preserve_subdirs}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Output directory
        if output_dir is None:
            output_dir = os.path.join(path_config.project_root, 'data', 'inference_results')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo các thư mục con chính
        self.masks_dir = self.output_dir / 'masks'
        self.metadata_dir = self.output_dir / 'metadata'
        self.visualizations_dir = self.output_dir / 'visualizations'
        
        self.masks_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"   ├── masks/")
        print(f"   ├── metadata/")
        print(f"   └── visualizations/")
    
    def _load_model(self, model_path: str) -> SegmentationModel:
        """Load trained segmentation model"""
        model = SegmentationModel(
            encoder_name=seg_config.encoder_name,
            num_classes=seg_config.num_classes,
            in_channels=seg_config.in_channels
        )
        
        with torch.serialization.safe_globals([np.dtype, np.float32, np.int64, np.void]):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Tiền xử lý ảnh giống với khi training
        
        Returns:
            input_tensor: (1, 1, H, W)
            original_size: (H, W)
        """
        # Đọc ảnh
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_h, original_w = image.shape[:2]
        
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize về kích thước input của model
        input_h, input_w = seg_config.input_size
        resized = cv2.resize(gray, (input_w, input_h))
        
        # Normalize
        from src.config import preprocess_config
        normalized = (resized.astype(np.float32) / 255.0 - preprocess_config.mean) / preprocess_config.std
        
        # Convert to tensor
        input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor, (original_h, original_w)
    
    def predict_mask(self, image_path: Path) -> Dict:
        """
        Dự đoán mask cho một ảnh
        
        Returns:
            Dict: {
                'mask': np.ndarray (H, W),
                'prob': np.ndarray (H, W),
                'defects': list of defect info,
                'num_defects': int
            }
        """
        input_tensor, (orig_h, orig_w) = self.preprocess_image(image_path)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
        
        # Resize về kích thước gốc
        mask = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Trích xuất đặc trưng lỗi
        defects_info = extract_defect_features(binary_mask)
        
        return {
            'mask': binary_mask,
            'prob': mask,
            'defects': defects_info.get('defects', []),
            'num_defects': defects_info.get('num_defects', 0),
            'original_size': (orig_h, orig_w)
        }
    
    def _get_output_paths(self, image_path: Path, subdir_name: str = None):
        """
        Lấy đường dẫn output cho các file
        
        Args:
            image_path: Đường dẫn ảnh gốc
            subdir_name: Tên thư mục con (nếu có)
        
        Returns:
            Tuple[Path, Path, Path]: (mask_path, metadata_path, vis_path)
        """
        stem = image_path.stem
        
        if self.preserve_subdirs and subdir_name:
            # Lưu theo cấu trúc thư mục con: masks/subdir_name/file_mask.png
            mask_subdir = self.masks_dir / subdir_name
            metadata_subdir = self.metadata_dir / subdir_name
            vis_subdir = self.visualizations_dir / subdir_name
            
            mask_subdir.mkdir(parents=True, exist_ok=True)
            metadata_subdir.mkdir(parents=True, exist_ok=True)
            vis_subdir.mkdir(parents=True, exist_ok=True)
            
            mask_path = mask_subdir / f"{stem}_mask.png"
            metadata_path = metadata_subdir / f"{stem}_metadata.json"
            vis_path = vis_subdir / f"{stem}_vis.png"
        else:
            # Lưu gộp vào thư mục chính
            mask_path = self.masks_dir / f"{stem}_mask.png"
            metadata_path = self.metadata_dir / f"{stem}_metadata.json"
            vis_path = self.visualizations_dir / f"{stem}_vis.png"
        
        return mask_path, metadata_path, vis_path
    
    def process_single_image(self, 
                             image_path: Path,
                             subdir_name: str = None,
                             save_results: bool = True,
                             visualize: bool = False) -> Dict:
        """
        Xử lý một ảnh đơn lẻ
        
        Args:
            image_path: Đường dẫn ảnh
            subdir_name: Tên thư mục con (ví dụ: 'T1_S148_I108_1')
            save_results: Có lưu kết quả không
            visualize: Có tạo visualization không
        """
        print(f"📷 Processing: {subdir_name}/{image_path.name}" if subdir_name else f"📷 Processing: {image_path.name}")
        
        # Predict
        result = self.predict_mask(image_path)
        
        if save_results:
            # Lấy đường dẫn output
            mask_path, metadata_path, vis_path = self._get_output_paths(image_path, subdir_name)
            
            # Lưu mask
            cv2.imwrite(str(mask_path), result['mask'] * 255)
            
            # Lưu probability map
            prob_path = mask_path.parent / f"{image_path.stem}_prob.png"
            prob_vis = (result['prob'] * 255).astype(np.uint8)
            cv2.imwrite(str(prob_path), prob_vis)
            
            # Lưu metadata
            metadata = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'subdir': subdir_name,
                'original_size': result['original_size'],
                'num_defects': result['num_defects'],
                'defects': result['defects'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Saved mask to: {mask_path}")
            print(f"   ✅ Saved metadata to: {metadata_path}")
            print(f"   📊 Found {result['num_defects']} defects")
        
        if visualize:
            self._visualize_result(image_path, result, subdir_name)
        
        return result
    
    def _visualize_result(self, image_path: Path, result: Dict, subdir_name: str = None):
        """Visualize segmentation result"""
        import matplotlib.pyplot as plt
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(result['mask'], cmap='gray')
        axes[1].set_title(f'Predicted Mask ({result["num_defects"]} defects)')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        color_mask = np.zeros_like(image)
        color_mask[result['mask'] == 1] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
        
        # Draw bounding boxes
        for i, defect in enumerate(result['defects']):
            x, y, w, h = defect['bbox']
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, f"#{i+1}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay with Defects')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Lưu visualization
        _, _, vis_path = self._get_output_paths(image_path, subdir_name)
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   🖼️ Saved visualization to: {vis_path}")
    
    def process_folder(self, 
                       folder_path: Path,
                       recursive: bool = True) -> pd.DataFrame:
        """
        Xử lý toàn bộ ảnh trong một thư mục (giữ nguyên cấu trúc thư mục con)
        
        Args:
            folder_path: Đường dẫn đến thư mục chứa ảnh (có thể có thư mục con)
            recursive: Có duyệt thư mục con không
        
        Returns:
            pd.DataFrame: Tổng hợp kết quả
        """
        print(f"\n{'='*60}")
        print(f"🚀 BẮT ĐẦU XỬ LÝ THƯ MỤC")
        print(f"{'='*60}")
        print(f"📁 Folder: {folder_path}")
        print(f"{'='*60}\n")
        
        results = []
        success_count = 0
        fail_count = 0
        
        if recursive:
            # Duyệt tất cả thư mục con
            subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
            
            if not subdirs:
                # Không có thư mục con, xử lý file trực tiếp trong folder
                return self._process_images_in_folder(folder_path, None, results, success_count, fail_count)
            
            for subdir in subdirs:
                print(f"\n📁 Processing subdirectory: {subdir.name}")
                results, success_count, fail_count = self._process_images_in_folder(
                    subdir, subdir.name, results, success_count, fail_count
                )
        else:
            # Chỉ xử lý file trực tiếp trong folder, không duyệt con
            results, success_count, fail_count = self._process_images_in_folder(
                folder_path, None, results, success_count, fail_count
            )
        
        # Lưu tổng kết
        df = pd.DataFrame(results)
        summary_path = self.output_dir / 'processing_summary.csv'
        df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"📊 KẾT QUẢ XỬ LÝ")
        print(f"{'='*60}")
        print(f"✅ Thành công: {success_count}/{success_count + fail_count}")
        print(f"❌ Thất bại: {fail_count}/{success_count + fail_count}")
        print(f"💾 Summary saved to: {summary_path}")
        
        # Tạo báo cáo tổng hợp metadata
        self._create_combined_metadata()
        
        return df
    
    def _process_images_in_folder(self, 
                                   folder_path: Path, 
                                   subdir_name: str,
                                   results: List,
                                   success_count: int,
                                   fail_count: int) -> Tuple[List, int, int]:
        """
        Xử lý tất cả ảnh trong một thư mục cụ thể
        """
        # extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        # images = []
        # for ext in extensions:
        #     images.extend(folder_path.glob(f"*{ext}"))

        images = list(folder_path.glob("*.png"))
        
        print(f"   📸 Found {len(images)} images")
        
        for img_path in tqdm(images, desc=f"   Processing", leave=False):
            try:
                result = self.process_single_image(img_path, subdir_name, save_results=True, visualize=True)
                results.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'subdir': subdir_name,
                    'num_defects': result['num_defects'],
                    'mask_path': str(self.masks_dir / subdir_name / f"{img_path.stem}_mask.png") if subdir_name else str(self.masks_dir / f"{img_path.stem}_mask.png"),
                    'metadata_path': str(self.metadata_dir / subdir_name / f"{img_path.stem}_metadata.json") if subdir_name else str(self.metadata_dir / f"{img_path.stem}_metadata.json"),
                    'status': 'success'
                })
                success_count += 1
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'subdir': subdir_name,
                    'status': 'failed',
                    'error': str(e)
                })
                fail_count += 1
        
        return results, success_count, fail_count
    
    def process_from_dataframe(self, 
                               df: pd.DataFrame,
                               image_root: Path,
                               filename_col: str = 'FileName') -> pd.DataFrame:
        """
        Xử lý ảnh từ DataFrame (giống cấu trúc preprocessing)
        Giữ nguyên thông tin subset_name làm thư mục con
        """
        print(f"\n{'='*60}")
        print(f"🚀 BẮT ĐẦU XỬ LÝ TỪ DATAFRAME")
        print(f"{'='*60}")
        print(f"📊 DataFrame có {len(df)} dòng")
        print(f"📁 Image root: {image_root}")
        print(f"{'='*60}\n")
        
        results = []
        success_count = 0
        fail_count = 0
        
        # Group by subset_name (thư mục con)
        for subset_name, group_df in df.groupby('subset_name'):
            print(f"\n📁 Processing subset: {subset_name}")
            
            for idx, row in tqdm(group_df.iterrows(), total=len(group_df), desc=f"   {subset_name}"):
                img_name = row[filename_col]
                img_path = image_root / subset_name / img_name
                
                if not img_path.exists():
                    # Thử tìm trong thư mục gốc
                    alt_path = image_root / img_name
                    if alt_path.exists():
                        img_path = alt_path
                    else:
                        print(f"   ⚠️ Image not found: {img_path}")
                        results.append({
                            'image_name': img_name,
                            'subset': subset_name,
                            'status': 'missing',
                            'error': 'File not found'
                        })
                        fail_count += 1
                        continue
                
                try:
                    result = self.process_single_image(img_path, subset_name, save_results=True, visualize=True)
                    results.append({
                        'image_name': img_name,
                        'subset': subset_name,
                        'original_name': row.get('FileName', img_name),
                        'num_defects': result['num_defects'],
                        'defects': json.dumps(result['defects']),
                        'mask_path': str(self.masks_dir / subset_name / f"{img_path.stem}_mask.png"),
                        'metadata_path': str(self.metadata_dir / subset_name / f"{img_path.stem}_metadata.json"),
                        'status': 'success'
                    })
                    success_count += 1
                except Exception as e:
                    print(f"   ❌ Error processing {img_name}: {e}")
                    results.append({
                        'image_name': img_name,
                        'subset': subset_name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    fail_count += 1
        
        result_df = pd.DataFrame(results)
        summary_path = self.output_dir / 'dataframe_processing_summary.csv'
        result_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"📊 KẾT QUẢ XỬ LÝ")
        print(f"{'='*60}")
        print(f"✅ Thành công: {success_count}/{success_count + fail_count}")
        print(f"❌ Thất bại: {fail_count}/{success_count + fail_count}")
        print(f"💾 Summary saved to: {summary_path}")
        
        # Tạo báo cáo tổng hợp
        self._create_combined_metadata()
        
        return result_df
    
    def _create_combined_metadata(self):
        """Tạo file metadata tổng hợp từ tất cả kết quả"""
        all_metadata = []
        
        # Đọc metadata từ cấu trúc thư mục con
        if self.preserve_subdirs:
            # Duyệt tất cả thư mục con trong metadata_dir
            for subdir in self.metadata_dir.iterdir():
                if subdir.is_dir():
                    for meta_file in subdir.glob("*_metadata.json"):
                        try:
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                all_metadata.append(data)
                        except Exception as e:
                            print(f"⚠️ Error reading {meta_file}: {e}")
        else:
            # Đọc từ thư mục chính
            for meta_file in self.metadata_dir.glob("*_metadata.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_metadata.append(data)
                except Exception as e:
                    print(f"⚠️ Error reading {meta_file}: {e}")
        
        if all_metadata:
            combined_path = self.output_dir / 'all_metadata.json'
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)
            
            # Thống kê theo thư mục con
            subdir_stats = {}
            for m in all_metadata:
                subdir = m.get('subdir', 'root')
                subdir_stats[subdir] = subdir_stats.get(subdir, 0) + 1
            
            total_defects = sum(m.get('num_defects', 0) for m in all_metadata)
            avg_defects = total_defects / len(all_metadata) if all_metadata else 0
            
            print(f"\n📊 TỔNG HỢP METADATA")
            print(f"   📸 Tổng số ảnh: {len(all_metadata)}")
            print(f"   🕳️ Tổng số lỗi: {total_defects}")
            print(f"   📈 Trung bình lỗi/ảnh: {avg_defects:.2f}")
            print(f"\n   📁 Phân bố theo thư mục:")
            for subdir, count in sorted(subdir_stats.items()):
                print(f"      - {subdir}: {count} ảnh")
            print(f"   💾 Combined metadata: {combined_path}")


def main():
    """Main function"""
    print("=" * 60)
    print("🔬 BATCH SEGMENTATION INFERENCE")
    print("=" * 60)
    
    # Đường dẫn model đã fine-tune
    MODEL_PATH = os.path.join(path_config.segmentation_models, 'best_model.pth')
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Please train the model first: python scripts/02_train_segmentation.py")
        return
    
    # Khởi tạo runner
    runner = BatchSegmentationRunner(
        model_path=MODEL_PATH,
        device='cpu',  # hoặc 'cuda' nếu có GPU
        output_dir='./data/inference_results',
        preserve_subdirs=True  # Giữ nguyên cấu trúc thư mục con
    )
    
    # Xử lý toàn bộ thư mục segmentation đã preprocess
    SEGMENTATION_DIR = Path(path_config.data_preprocessed) / 'segmentation'
    
    if SEGMENTATION_DIR.exists():
        print(f"\n📁 Processing all images in: {SEGMENTATION_DIR}")
        
        # Xử lý toàn bộ thư mục (tự động duyệt thư mục con)
        df = runner.process_folder(SEGMENTATION_DIR, recursive=True)
        print(f"\n✅ Final results saved to: {runner.output_dir}")
    else:
        print(f"⚠️ Segmentation directory not found: {SEGMENTATION_DIR}")


if __name__ == "__main__":
    main()