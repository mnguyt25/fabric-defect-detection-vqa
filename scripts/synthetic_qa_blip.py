#!/usr/bin/env python
"""
Script: Sinh dữ liệu câu hỏi - trả lời bằng BLIP (base model)
Sử dụng ảnh tiền xử lý (grayscale) từ thư mục preprocessed/segmentation
Ảnh được chuyển sang RGB cho BLIP
Model: Salesforce/blip-vqa-base (nhẹ hơn, chạy tốt trên CPU)
"""
import os
import time
import json
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import BlipProcessor, BlipForQuestionAnswering


class BlipQABatchProcessor:
    """
    Xử lý QA với BLIP (base model), sử dụng ảnh tiền xử lý (grayscale)
    Mỗi subset là một batch riêng
    Kết quả được lưu theo cấu trúc tương ứng
    """
    
    def __init__(self, 
                 input_root: Path,
                 output_root: Path = None,
                 target_size: tuple = (224, 224),
                 questions_per_image: int = 5,
                 device: str = None,
                 model_name: str = "Salesforce/blip-vqa-base",
                 delay_between_requests: float = 0.5,
                 resume: bool = True):
        """
        Khởi tạo processor
        
        Args:
            input_root: Thư mục gốc chứa ảnh tiền xử lý (preprocessed/segmentation)
            output_root: Thư mục gốc lưu kết quả (mặc định: thư mục cha của input_root / "blip_qa_results")
            target_size: Kích thước resize cho ảnh (width, height)
            questions_per_image: Số câu hỏi sinh cho mỗi ảnh
            device: 'cuda' hoặc 'cpu' (None = tự động chọn)
            model_name: Tên model BLIP
            delay_between_requests: Delay giữa các request (giây)
            resume: Tiếp tục từ lần chạy trước (bỏ qua ảnh đã xử lý)
        """
        self.input_root = Path(input_root)
        self.target_size = target_size
        
        # Tạo output root cùng cấp với segmentation (trong thư mục preprocessed)
        if output_root is None:
            # Lấy thư mục cha của input_root (data/preprocessed)
            parent_dir = self.input_root.parent
            output_root = parent_dir / "blip_qa_results"
        self.output_root = Path(output_root)
        
        self.questions_per_image = questions_per_image
        self.delay_between_requests = delay_between_requests
        self.resume = resume
        
        # Thiết lập device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"📁 Input root: {self.input_root}")
        print(f"📁 Output root: {self.output_root}")
        print(f"📐 Target size: {self.target_size}")
        print(f"💻 Device: {self.device}")
        print(f"🤖 Model: {model_name}")
        print(f"🔄 Resume mode: {self.resume}")
        
        # Tạo thư mục output gốc
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load BLIP model và processor"""
        print("\n🚀 Đang tải BLIP model...")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!\n")
    
    def _get_questions(self) -> List[str]:
        """Lấy danh sách câu hỏi mẫu cho mỗi ảnh"""
        questions = [
            "How many defects are visible on this fabric?",
            "What type of defect is this?",
            "Where is the defect located?",
            "How severe is this defect?",
            "What is the condition of this fabric?"
        ]
        return questions[:self.questions_per_image]
    
    def _load_and_preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Đọc ảnh grayscale từ thư mục preprocessed, chuyển thành RGB cho BLIP
        
        Args:
            image_path: Đường dẫn đến ảnh grayscale (đã qua tiền xử lý)
        
        Returns:
            PIL Image ở định dạng RGB
        """
        try:
            # Đọc ảnh grayscale
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"   ⚠️ Không thể đọc ảnh: {image_path}")
                return None
            
            # Resize về kích thước target
            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Chuyển grayscale (1 kênh) thành RGB (3 kênh)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # Chuyển thành PIL Image
            pil_image = Image.fromarray(img_rgb)
            
            return pil_image
            
        except Exception as e:
            print(f"   ❌ Lỗi xử lý ảnh: {e}")
            return None
    
    def generate_qa_for_image(self, image_path: Path) -> Optional[List[Dict]]:
        """
        Sinh câu hỏi - trả lời cho một ảnh tiền xử lý
        
        Returns:
            List[Dict]: Danh sách các cặp QA
        """
        try:
            # Load và preprocess ảnh
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return None
            
            questions = self._get_questions()
            results = []
            
            for question in questions:
                # BLIP sử dụng inputs trực tiếp không cần prompt đặc biệt
                inputs = self.processor(image, question, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=3,
                        early_stopping=True
                    )
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                results.append({
                    "question": question,
                    "answer": answer
                })
            
            return results
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            return None
    
    def process_subset(self, subset_name: str, image_folder: Path) -> Dict:
        """
        Xử lý một subset (batch)
        
        Args:
            subset_name: Tên subset (ví dụ: T1_S148_I108_1)
            image_folder: Thư mục chứa ảnh tiền xử lý của subset này
        """
        # Tạo thư mục output cho subset này (giữ nguyên cấu trúc như segmentation)
        subset_output_dir = self.output_root / subset_name
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Đọc danh sách ảnh (chỉ .png vì ảnh preprocessed là .png)
        images = list(image_folder.glob("*.png"))
        
        # Thêm cả .jpg nếu có
        images.extend(image_folder.glob("*.jpg"))
        images.extend(image_folder.glob("*.jpeg"))
        
        # Loại bỏ trùng lặp và sắp xếp
        images = list(set(images))
        images.sort()
        
        print(f"\n{'='*60}")
        print(f"🚀 XỬ LÝ SUBSET: {subset_name}")
        print(f"{'='*60}")
        print(f"📸 Số ảnh: {len(images)}")
        print(f"📂 Thư mục ảnh: {image_folder}")
        print(f"📁 Thư mục output: {subset_output_dir}")
        print(f"📐 Kích thước ảnh sau preprocess: {self.target_size}")
        
        # Kiểm tra ảnh đã xử lý (resume mode)
        processed_images = set()
        if self.resume:
            for qa_file in subset_output_dir.glob("*_qa.json"):
                processed_images.add(qa_file.stem.replace('_qa', ''))
        
        if processed_images:
            print(f"🔄 Resume mode: {len(processed_images)} ảnh đã xử lý, bỏ qua")
        
        results = []
        success_count = 0
        fail_count = 0
        skipped_count = 0
        total_qa_pairs = 0
        
        for idx, img_path in enumerate(tqdm(images, desc=f"   {subset_name}")):
            # Kiểm tra đã xử lý chưa
            if img_path.stem in processed_images:
                skipped_count += 1
                continue
            
            print(f"\n📷 [{idx+1}/{len(images)}] {img_path.name}")
            
            qa_pairs = self.generate_qa_for_image(img_path)
            
            if qa_pairs and len(qa_pairs) > 0:
                # Format conversations cho BLIP fine-tuning
                conversations = []
                for qa in qa_pairs:
                    conversations.append({
                        "from": "human",
                        "value": qa['question']
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": qa['answer']
                    })
                
                result_item = {
                    "image": str(img_path),
                    "image_id": img_path.stem,
                    "subset": subset_name,
                    "target_size": self.target_size,
                    "conversations": conversations,
                    "qa_pairs": qa_pairs,
                    "num_qa_pairs": len(qa_pairs),
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result_item)
                success_count += 1
                total_qa_pairs += len(qa_pairs)
                
                # Lưu kết quả từng ảnh
                img_qa_file = subset_output_dir / f"{img_path.stem}_qa.json"
                with open(img_qa_file, 'w', encoding='utf-8') as f:
                    json.dump(result_item, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Thành công: {len(qa_pairs)} câu hỏi")
                print(f"   💾 Lưu tại: {img_qa_file}")
            else:
                fail_count += 1
                print(f"   ❌ Thất bại")
            
            # Delay nhẹ để tránh quá tải
            if idx < len(images) - 1:
                time.sleep(self.delay_between_requests)
        
        # Lưu tổng kết subset
        subset_summary = {
            'subset_name': subset_name,
            'total_images': len(images),
            'success_count': success_count,
            'fail_count': fail_count,
            'skipped_count': skipped_count,
            'total_qa_pairs': total_qa_pairs,
            'questions_per_image': self.questions_per_image,
            'target_size': self.target_size,
            'model': "BLIP",
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = subset_output_dir / "subset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(subset_summary, f, indent=2, ensure_ascii=False)
        
        # Lưu dạng JSONL cho fine-tuning (riêng từng subset)
        jsonl_file = subset_output_dir / f"{subset_name}_blip.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for result in results:
                blip_item = {
                    "image": result['image'],
                    "conversations": result['conversations']
                }
                f.write(json.dumps(blip_item, ensure_ascii=False) + '\n')
        
        # Lưu CSV (riêng từng subset)
        csv_data = []
        for result in results:
            for qa in result.get('qa_pairs', []):
                csv_data.append({
                    'subset': subset_name,
                    'image': Path(result['image']).name,
                    'image_stem': Path(result['image']).stem,
                    'question': qa['question'],
                    'answer': qa['answer']
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = subset_output_dir / f"{subset_name}_qa_dataset.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"📊 KẾT QUẢ SUBSET {subset_name}")
        print(f"{'='*60}")
        print(f"✅ Thành công: {success_count}/{len(images)}")
        print(f"❌ Thất bại: {fail_count}/{len(images)}")
        print(f"⏭️ Bỏ qua (đã xử lý): {skipped_count}")
        print(f"📝 QA pairs: {total_qa_pairs}")
        print(f"📁 Thư mục kết quả: {subset_output_dir}")
        
        return subset_summary
    
    def process_all_subsets(self, subset_names: List[str] = None, max_images_per_subset: int = None):
        """
        Xử lý tất cả subsets
        
        Args:
            subset_names: Danh sách tên subset cần xử lý (None = tất cả)
            max_images_per_subset: Giới hạn số ảnh mỗi subset (None = không giới hạn)
        """
        # Tìm tất cả subsets trong input_root
        if subset_names is None:
            subsets = [d for d in self.input_root.iterdir() if d.is_dir()]
            # Loại trừ thư mục output (nếu nó nằm trong input_root)
            subsets = [d for d in subsets if d.name != self.output_root.name]
            subset_names = [d.name for d in subsets]
        
        print(f"\n📦 Tìm thấy {len(subset_names)} subsets: {subset_names}")
        
        if len(subset_names) == 0:
            print("⚠️ Không có subset nào để xử lý!")
            return {}
        
        print(f"\n{'='*60}")
        print(f"🚀 XỬ LÝ {len(subset_names)} SUBSETS")
        print(f"{'='*60}")
        
        all_results = {}
        total_success = 0
        total_images = 0
        total_qa = 0
        
        for subset_name in subset_names:
            image_folder = self.input_root / subset_name
            
            if not image_folder.exists():
                print(f"⚠️ Không tìm thấy thư mục: {image_folder}")
                continue
            
            # Giới hạn số ảnh nếu cần
            if max_images_per_subset:
                all_images = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))
                if len(all_images) > max_images_per_subset:
                    print(f"⚠️ Subset {subset_name} có {len(all_images)} ảnh, chỉ xử lý {max_images_per_subset} ảnh đầu tiên")
                    import tempfile
                    import shutil
                    temp_dir = Path(tempfile.mkdtemp())
                    for img in all_images[:max_images_per_subset]:
                        shutil.copy2(img, temp_dir / img.name)
                    image_folder = temp_dir
            
            result = self.process_subset(subset_name, image_folder)
            all_results[subset_name] = result
            
            total_success += result.get('success_count', 0)
            total_images += result.get('total_images', 0)
            total_qa += result.get('total_qa_pairs', 0)
        
        # Tạo master dataset
        self._create_master_dataset()
        
        print(f"\n{'='*60}")
        print(f"📊 TỔNG KẾT")
        print(f"{'='*60}")
        print(f"📸 Tổng số ảnh đã xử lý: {total_images}")
        print(f"✅ Thành công: {total_success}")
        print(f"📝 Tổng QA pairs: {total_qa}")
        print(f"📁 Kết quả lưu tại: {self.output_root}")
        
        return all_results
    
    def _create_master_dataset(self):
        """Tạo file tổng hợp tất cả QA pairs từ các subsets"""
        all_blip_data = []
        all_csv_data = []
        
        # Duyệt qua tất cả thư mục subset trong output_root
        for subset_dir in self.output_root.iterdir():
            if not subset_dir.is_dir():
                continue
            
            # Đọc JSONL
            jsonl_file = subset_dir / f"{subset_dir.name}_blip.jsonl"
            if jsonl_file.exists():
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                data['subset'] = subset_dir.name
                                all_blip_data.append(data)
                            except:
                                pass
            
            # Đọc CSV
            csv_file = subset_dir / f"{subset_dir.name}_qa_dataset.csv"
            if csv_file.exists():
                all_csv_data.append(pd.read_csv(csv_file))
        
        # Lưu master files
        if all_blip_data:
            master_jsonl = self.output_root / "master_blip_dataset.jsonl"
            with open(master_jsonl, 'w', encoding='utf-8') as f:
                for item in all_blip_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\n📊 Master JSONL: {master_jsonl} ({len(all_blip_data)} samples)")
            print(f"   Format: BLIP fine-tuning ready")
        
        if all_csv_data:
            master_df = pd.concat(all_csv_data, ignore_index=True)
            master_csv = self.output_root / "master_qa_dataset.csv"
            master_df.to_csv(master_csv, index=False)
            print(f"📊 Master CSV: {master_csv} ({len(master_df)} QA pairs)")
        
        # Tạo file thống kê
        stats = {
            'model': 'BLIP',
            'target_size': self.target_size,
            'total_subsets': len([d for d in self.output_root.iterdir() if d.is_dir()]),
            'total_samples': len(all_blip_data),
            'total_qa_pairs': len(pd.concat(all_csv_data, ignore_index=True)) if all_csv_data else 0,
            'timestamp': datetime.now().isoformat(),
            'questions_per_image': self.questions_per_image,
            'subsets': [d.name for d in self.output_root.iterdir() if d.is_dir()]
        }
        
        stats_file = self.output_root / "generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"📊 Stats file: {stats_file}")


# ========== SỬ DỤNG ==========

if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa ảnh tiền xử lý
    # Cấu trúc: preprocessed/segmentation/subset_name/*.png
    INPUT_ROOT = r"D:\Study\On-going\Xu_ly_anh\Computer Vision\defect_detection\data\preprocessed\segmentation"
    
    # Khởi tạo processor - output root là blip_qa_results cùng cấp với segmentation
    processor = BlipQABatchProcessor(
        input_root=Path(INPUT_ROOT),
        output_root=None,  # Tự động tạo blip_qa_results trong thư mục preprocessed
        target_size=(224, 224),
        questions_per_image=5,
        device="cpu",  # hoặc "cuda" nếu có GPU
        model_name="Salesforce/blip-vqa-base",
        delay_between_requests=0.5,
        resume=True
    )
    
    # Xử lý tất cả subsets
    processor.process_all_subsets(
        subset_names=None,  # None = xử lý tất cả
        max_images_per_subset=None  # Không giới hạn
    )