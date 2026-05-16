#!/usr/bin/env python
"""
Script: Sinh dữ liệu câu hỏi - trả lời bằng Gemini API cho SmolVLM
Sử dụng ảnh đã trích xuất từ extract_raw_images
Tự động chia nhỏ subset có >1500 ảnh để tránh quota
Kết quả được gộp về đúng subset gốc
"""
import os
import time
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from google import genai
from google.genai import types
import pandas as pd
from tqdm import tqdm
import random


class MultiKeyGeminiManager:
    """
    Quản lý nhiều API keys cho Gemini, tự động chuyển đổi khi hết quota
    Hỗ trợ round-robin và load balancing
    """
    def __init__(self, api_keys: List[str], keys_config_file: str = "api_keys_state.json"):
        self.api_keys = api_keys
        self.keys_config_file = keys_config_file
        self.current_key_index = 0
        self.key_stats = self.load_key_stats()
        
    def load_key_stats(self) -> Dict:
        if os.path.exists(self.keys_config_file):
            with open(self.keys_config_file, 'r') as f:
                return json.load(f)
        else:
            stats = {}
            for i, key in enumerate(self.api_keys):
                key_mask = f"{key[:8]}...{key[-8:]}" if len(key) > 16 else f"Key_{i}"
                stats[key_mask] = {
                    'original_key': key,
                    'requests_today': 0,
                    'last_reset_date': datetime.now().strftime("%Y-%m-%d"),
                    'status': 'active',
                    'error_count': 0
                }
            return stats
    
    def save_key_stats(self):
        with open(self.keys_config_file, 'w') as f:
            json.dump(self.key_stats, f, indent=2)
    
    def reset_daily_counters(self):
        today = datetime.now().strftime("%Y-%m-%d")
        for key_mask, stats in self.key_stats.items():
            if stats['last_reset_date'] != today:
                stats['requests_today'] = 0
                stats['last_reset_date'] = today
                stats['status'] = 'active'
        self.save_key_stats()
    
    def get_available_key_sequential(self) -> Optional[str]:
        """Lấy key theo thứ tự (sequential) - cách cũ"""
        self.reset_daily_counters()
        for key_mask, stats in self.key_stats.items():
            if stats['requests_today'] < 1500 and stats['status'] == 'active':
                return stats['original_key']
        return None
    
    def get_available_key_round_robin(self) -> Optional[str]:
        """
        Lấy key theo vòng tròn (round-robin) để phân bổ đều tải
        """
        self.reset_daily_counters()
        
        # Thu thập tất cả key còn quota
        available_keys = []
        for key_mask, stats in self.key_stats.items():
            if stats['requests_today'] < 1500 and stats['status'] == 'active':
                available_keys.append(stats['original_key'])
        
        if not available_keys:
            print("\n⚠️ Tất cả API keys đã đạt giới hạn 1500 requests/ngày!")
            return None
        
        # Lấy key theo vòng tròn
        key = available_keys[self.current_key_index % len(available_keys)]
        self.current_key_index += 1
        return key
    
    def get_available_key_load_balanced(self) -> Optional[str]:
        """
        Lấy key có số request ít nhất (load balancing)
        """
        self.reset_daily_counters()
        
        best_key = None
        min_requests = float('inf')
        
        for key_mask, stats in self.key_stats.items():
            if stats['status'] != 'active':
                continue
            if stats['requests_today'] < min_requests:
                min_requests = stats['requests_today']
                best_key = stats['original_key']
        
        if best_key is None:
            print("\n⚠️ Không có API key khả dụng nào!")
        
        return best_key
    
    def get_available_key(self) -> Optional[str]:
        """Mặc định dùng round-robin (có thể đổi thành load_balanced)"""
        return self.get_available_key_round_robin()
    
    def mark_request_used(self, api_key: str):
        for key_mask, stats in self.key_stats.items():
            if stats['original_key'] == api_key:
                stats['requests_today'] += 1
                self.save_key_stats()
                remaining = 1500 - stats['requests_today']
                print(f"📊 Key {key_mask}: {stats['requests_today']}/1500 (còn {remaining})")
                if remaining == 0:
                    print(f"⚠️ Key {key_mask} đã hết quota!")
                break
    
    def mark_key_error(self, api_key: str, error_msg: str):
        for key_mask, stats in self.key_stats.items():
            if stats['original_key'] == api_key:
                stats['error_count'] += 1
                stats['status'] = 'error'
                print(f"❌ Key {key_mask} gặp lỗi: {error_msg}")
                self.save_key_stats()
                break
    
    def get_status_report(self) -> str:
        self.reset_daily_counters()
        report = "\n" + "="*50 + "\n"
        report += "📊 BÁO CÁO SỬ DỤNG API KEYS\n"
        report += "="*50 + "\n"
        for key_mask, stats in self.key_stats.items():
            remaining = 1500 - stats['requests_today']
            status_emoji = "✅" if stats['status'] == 'active' else "❌"
            report += f"{status_emoji} {key_mask}: {stats['requests_today']}/1500 (còn {remaining})\n"
        report += "="*50
        return report


class GeminiQABatchProcessor:
    """
    Xử lý batch QA với Gemini API, tự động chia nhỏ subset có >1500 ảnh
    Kết quả được gộp về đúng subset gốc
    """
    
    def __init__(self, 
                 input_root: Path,
                 api_keys: List[str],
                 output_root: Path = None,
                 questions_per_image: int = 5,
                 delay_between_requests: float = 2.0,
                 resume: bool = True,
                 max_retries: int = 3):
        """
        Khởi tạo processor
        
        Args:
            input_root: Thư mục gốc chứa ảnh đã trích xuất (extracted_raw_images)
            api_keys: Danh sách API keys Gemini
            output_root: Thư mục gốc lưu kết quả (mặc định: input_root / "gemini_qa_results")
            questions_per_image: Số câu hỏi sinh cho mỗi ảnh
            delay_between_requests: Delay giữa các request (giây)
            resume: Tiếp tục từ lần chạy trước
            max_retries: Số lần thử lại khi gặp lỗi
        """
        self.input_root = Path(input_root)
        
        if output_root is None:
            output_root = self.input_root / "gemini_qa_results"
        self.output_root = Path(output_root)
        
        self.questions_per_image = questions_per_image
        self.delay_between_requests = delay_between_requests
        self.resume = resume
        self.max_retries = max_retries
        
        self.key_manager = MultiKeyGeminiManager(api_keys)
        
        # Tạo thư mục output
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Input root: {self.input_root}")
        print(f"📁 Output root: {self.output_root}")
        print(f"📝 Questions per image: {self.questions_per_image}")
        print(f"🔄 Resume mode: {self.resume}")
        print(f"🔄 Max retries: {self.max_retries}")
        print(f"⏱️ Delay between requests: {self.delay_between_requests}s")
    
    def _split_subset_if_needed(self, subset_name: str, image_folder: Path, max_images: int = 1500) -> List[Tuple[str, Path]]:
        """
        Kiểm tra và chia nhỏ subset nếu số ảnh > max_images
        
        Args:
            subset_name: Tên subset gốc
            image_folder: Thư mục chứa ảnh
            max_images: Số ảnh tối đa cho mỗi phần
        
        Returns:
            List[Tuple[str, Path]]: Danh sách (tên_phần, đường_dẫn_thư_mục_ảnh)
        """
        # Đếm số ảnh
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(image_folder.glob(ext))
            # Cũng tìm trong thư mục con
            for subdir in image_folder.iterdir():
                if subdir.is_dir():
                    images.extend(subdir.glob(ext))
        
        images = list(set(images))
        images.sort()
        
        if len(images) <= max_images:
            return [(subset_name, image_folder)]
        
        # Chia nhỏ thành nhiều phần
        num_parts = (len(images) + max_images - 1) // max_images
        parts = []
        
        # Tạo thư mục tạm thời cho các phần
        temp_base = self.input_root / f"_temp_split_{subset_name}"
        temp_base.mkdir(parents=True, exist_ok=True)
        
        for part_idx in range(num_parts):
            start = part_idx * max_images
            end = min(start + max_images, len(images))
            part_images = images[start:end]
            
            part_name = f"{subset_name}_part{part_idx + 1:02d}"
            part_folder = temp_base / part_name
            part_folder.mkdir(exist_ok=True)
            
            # Copy ảnh vào thư mục phần
            for img_path in part_images:
                dest_path = part_folder / img_path.name
                shutil.copy2(img_path, dest_path)
            
            parts.append((part_name, part_folder))
            print(f"   📦 Chia {subset_name}: phần {part_idx + 1}/{num_parts} có {len(part_images)} ảnh")
        
        return parts
    
    def _cleanup_temp_folders(self):
        """Dọn dẹp các thư mục tạm thời"""
        for temp_dir in self.input_root.glob("_temp_split_*"):
            shutil.rmtree(temp_dir)
            print(f"🗑️ Đã xóa thư mục tạm: {temp_dir}")
    
    def _build_prompt(self, num_questions: int) -> str:
        """Xây dựng prompt cho Gemini"""
        prompt = f"""You are a fabric defect detection expert. Analyze this fabric image and generate {num_questions} high-quality question-answer pairs for training a Vision Language Model (SmolVLM).

CRITICAL FORMAT REQUIREMENTS:
The output MUST be a JSON array with the following structure:

[
    {{"question": "Your question here...", "answer": "Your detailed answer here..."}},
    {{"question": "Another question...", "answer": "Another answer..."}}
]

RULES FOR QUESTIONS:
1. Questions should be diverse: defect type, location, severity, count, size, color, texture comparison
2. Include both simple (yes/no, count) and complex (describe, compare, explain) questions
3. Each question MUST be answerable from the image alone

RULES FOR ANSWERS:
1. Answers must be specific and fact-based on what you see
2. For yes/no questions: start with "Yes" or "No" then explain
3. For count questions: give exact number then describe
4. For location questions: specify coordinates or relative positions

OUTPUT ONLY THE JSON ARRAY, NO EXTRA TEXT, NO EXPLANATIONS.

Generate exactly {num_questions} question-answer pairs now:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Optional[List[Dict]]:
        """Parse Gemini response thành format QA pairs"""
        import re
        
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if not match:
            qa_pattern = r'"question"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]+)"'
            matches = re.findall(qa_pattern, response_text)
            if matches:
                return [{"question": q, "answer": a} for q, a in matches]
            return None
        
        try:
            data = json.loads(match.group())
            if isinstance(data, list) and len(data) > 0:
                normalized = []
                for item in data:
                    if 'question' in item and 'answer' in item:
                        normalized.append({
                            'question': item['question'].strip(),
                            'answer': item['answer'].strip()
                        })
                return normalized if normalized else data
        except json.JSONDecodeError:
            return None
        
        return None
    
    def process_single_image_with_retry(self, image_path: Path) -> Optional[Dict]:
        """
        Xử lý một ảnh đơn lẻ với cơ chế retry khi gặp lỗi
        """
        for attempt in range(self.max_retries):
            api_key = self.key_manager.get_available_key()
            if not api_key:
                return None
            
            # Thêm delay ngẫu nhiên để tránh rate limit
            if attempt > 0:
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"   🔄 Retry attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s...")
                time.sleep(wait_time)
            
            try:
                client = genai.Client(api_key=api_key)
                prompt = self._build_prompt(self.questions_per_image)
                
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt
                    ]
                )
                
                self.key_manager.mark_request_used(api_key)
                
                qa_pairs = self._parse_response(response.text)
                
                if qa_pairs and len(qa_pairs) > 0:
                    return {
                        'success': True,
                        'image': image_path.name,
                        'qa_pairs': qa_pairs,
                        'num_questions': len(qa_pairs),
                        'attempt': attempt + 1
                    }
                else:
                    return {
                        'success': False,
                        'image': image_path.name,
                        'error': 'Failed to parse JSON from response'
                    }
                
            except Exception as e:
                error_msg = str(e)
                
                # Nếu lỗi do quota hoặc rate limit, thử lại với key khác
                if "quota" in error_msg.lower() or "rate" in error_msg.lower() or "429" in error_msg:
                    print(f"   ⚠️ Rate limit/Quota error on attempt {attempt + 1}: {error_msg[:100]}")
                    self.key_manager.mark_key_error(api_key, error_msg)
                    continue  # Thử lại với key khác
                else:
                    # Lỗi khác, không retry
                    return {
                        'success': False,
                        'image': image_path.name,
                        'error': error_msg
                    }
        
        return {
            'success': False,
            'image': image_path.name,
            'error': f'Failed after {self.max_retries} retries'
        }
    
    def process_batch_part(self, part_name: str, image_folder: Path, original_subset: str) -> List[Dict]:
        """
        Xử lý một phần của subset (đã được chia nhỏ)
        Kết quả được lưu tạm để sau gộp vào subset gốc
        """
        # Đọc danh sách ảnh
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(image_folder.glob(ext))
        
        images = list(set(images))
        images.sort()
        
        print(f"\n{'='*60}")
        print(f"🚀 XỬ LÝ PHẦN: {part_name} (thuộc {original_subset})")
        print(f"{'='*60}")
        print(f"📸 Số ảnh: {len(images)}")
        print(self.key_manager.get_status_report())
        
        results = []
        success_count = 0
        fail_count = 0
        total_qa_pairs = 0
        
        # Thư mục lưu tạm kết quả cho phần này (sẽ gộp vào subset gốc sau)
        temp_part_dir = self.output_root / f"_temp_{part_name}"
        temp_part_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img_path in enumerate(tqdm(images, desc=f"   {part_name}")):
            print(f"\n📷 [{idx+1}/{len(images)}] {img_path.name}")
            
            # Thêm delay trước mỗi request
            time.sleep(random.uniform(self.delay_between_requests, self.delay_between_requests + 1))
            
            result = self.process_single_image_with_retry(img_path)
            
            if result and result.get('success'):
                results.append(result)
                success_count += 1
                total_qa_pairs += result.get('num_questions', 0)
                
                # Lưu kết quả tạm thời
                temp_item = {
                    "image": str(img_path),
                    "image_name": img_path.name,
                    "subset": original_subset,
                    "part": part_name,
                    "qa_pairs": result['qa_pairs'],
                    "attempt": result.get('attempt', 1),
                    "timestamp": datetime.now().isoformat()
                }
                
                temp_file = temp_part_dir / f"{img_path.stem}_temp.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(temp_item, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Thành công: {result['num_questions']} câu hỏi (attempt {result.get('attempt', 1)})")
            else:
                fail_count += 1
                error = result.get('error', 'Unknown') if result else "No API key"
                print(f"   ❌ Thất bại: {error}")
            
            if not self.key_manager.get_available_key():
                print(f"\n⚠️ Hết API key! Đã xử lý {success_count}/{len(images)} ảnh")
                break
        
        print(f"\n📊 KẾT QUẢ PHẦN {part_name}")
        print(f"   ✅ Thành công: {success_count}/{len(images)}")
        print(f"   📝 QA pairs: {total_qa_pairs}")
        
        return results
    
    def _merge_part_to_subset(self, subset_name: str):
        """
        Gộp tất cả các phần của một subset vào thư mục subset gốc
        """
        print(f"\n📦 Đang gộp kết quả cho subset: {subset_name}")
        
        # Tìm tất cả file tạm của subset này
        temp_files = list(self.output_root.glob(f"_temp_*_{subset_name}_part*/*_temp.json"))
        temp_files.extend(self.output_root.glob(f"_temp_{subset_name}_part*/*_temp.json"))
        
        if not temp_files:
            print(f"   Không có file tạm nào cho subset {subset_name}")
            return
        
        # Thư mục output cho subset này
        subset_output_dir = self.output_root / subset_name
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_smolvlm_data = []
        all_csv_data = []
        
        for temp_file in tqdm(temp_files, desc=f"   Gộp {subset_name}"):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Format cho SmolVLM
                    conversations = []
                    for qa in data.get('qa_pairs', []):
                        conversations.append({"from": "human", "value": qa['question']})
                        conversations.append({"from": "gpt", "value": qa['answer']})
                    
                    smolvlm_item = {
                        "image": data['image'],
                        "conversations": conversations
                    }
                    all_smolvlm_data.append(smolvlm_item)
                    
                    # Format cho CSV
                    for qa in data.get('qa_pairs', []):
                        all_csv_data.append({
                            'subset': subset_name,
                            'image': data['image_name'],
                            'question': qa['question'],
                            'answer': qa['answer']
                        })
                    
                    # Lưu file QA riêng cho từng ảnh
                    img_qa_file = subset_output_dir / f"{data['image_name'].replace('.', '_')}_qa.json"
                    with open(img_qa_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "image": data['image'],
                            "qa_pairs": data['qa_pairs'],
                            "timestamp": data['timestamp']
                        }, f, indent=2, ensure_ascii=False)
                        
            except Exception as e:
                print(f"   ⚠️ Lỗi đọc file {temp_file}: {e}")
        
        # Lưu JSONL cho subset
        if all_smolvlm_data:
            jsonl_file = subset_output_dir / f"{subset_name}_smolvlm.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for item in all_smolvlm_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"   ✅ JSONL: {jsonl_file} ({len(all_smolvlm_data)} samples)")
        
        # Lưu CSV cho subset
        if all_csv_data:
            df = pd.DataFrame(all_csv_data)
            csv_file = subset_output_dir / f"{subset_name}_qa_dataset.csv"
            df.to_csv(csv_file, index=False)
            print(f"   ✅ CSV: {csv_file} ({len(df)} QA pairs)")
        
        # Xóa thư mục tạm của các phần
        for temp_dir in self.output_root.glob(f"_temp_*{subset_name}*"):
            if temp_dir.is_dir():
                shutil.rmtree(temp_dir)
                print(f"   🗑️ Đã xóa thư mục tạm: {temp_dir}")
    
    def _merge_all_subsets(self):
        """
        Gộp tất cả kết quả từ các phần vào đúng subset gốc
        """
        print("\n" + "="*60)
        print("📦 ĐANG GỘP KẾT QUẢ VÀO CÁC SUBSET...")
        print("="*60)
        
        # Tìm tất cả subset đã được xử lý (qua các file tạm)
        processed_subsets = set()
        
        # Tìm từ pattern _temp_*_subsetname_part*
        for temp_dir in self.output_root.glob("_temp_*"):
            if temp_dir.is_dir():
                name = temp_dir.name.replace("_temp_", "")
                if "_part" in name:
                    subset_name = name.split("_part")[0]
                    processed_subsets.add(subset_name)
        
        # Cũng tìm từ pattern _temp_subsetname_part*
        for temp_dir in self.output_root.glob("_temp_*_part*"):
            if temp_dir.is_dir():
                name = temp_dir.name.replace("_temp_", "")
                if "_part" in name:
                    subset_name = name.split("_part")[0]
                    processed_subsets.add(subset_name)
        
        if not processed_subsets:
            print("⚠️ Không tìm thấy kết quả nào để gộp!")
            return
        
        print(f"📦 Các subset cần gộp: {sorted(processed_subsets)}")
        
        # Gộp từng subset
        for subset_name in sorted(processed_subsets):
            self._merge_part_to_subset(subset_name)
        
        # Tạo master dataset tổng hợp (tùy chọn)
        self._create_master_dataset()
    
    def _create_master_dataset(self):
        """
        Tạo file tổng hợp từ tất cả subsets (tùy chọn, vẫn giữ cấu trúc subset)
        """
        print("\n" + "="*60)
        print("📦 TẠO MASTER DATASET TỔNG HỢP...")
        print("="*60)
        
        all_smolvlm_data = []
        all_csv_data = []
        
        # Duyệt qua tất cả thư mục subset trong output_root
        for subset_dir in self.output_root.iterdir():
            if not subset_dir.is_dir():
                continue
            if subset_dir.name.startswith("_temp_"):
                continue
            
            # Đọc JSONL của subset
            jsonl_file = subset_dir / f"{subset_dir.name}_smolvlm.jsonl"
            if jsonl_file.exists():
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                data['subset'] = subset_dir.name
                                all_smolvlm_data.append(data)
                            except:
                                pass
            
            # Đọc CSV của subset
            csv_file = subset_dir / f"{subset_dir.name}_qa_dataset.csv"
            if csv_file.exists():
                all_csv_data.append(pd.read_csv(csv_file))
        
        # Lưu master files
        if all_smolvlm_data:
            master_jsonl = self.output_root / "master_smolvlm_dataset.jsonl"
            with open(master_jsonl, 'w', encoding='utf-8') as f:
                for item in all_smolvlm_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✅ Master JSONL: {master_jsonl} ({len(all_smolvlm_data)} samples)")
        
        if all_csv_data:
            master_df = pd.concat(all_csv_data, ignore_index=True)
            master_csv = self.output_root / "master_qa_dataset.csv"
            master_df.to_csv(master_csv, index=False)
            print(f"✅ Master CSV: {master_csv} ({len(master_df)} QA pairs)")
        
        # Lưu thống kê
        stats = {
            'total_subsets': len([d for d in self.output_root.iterdir() if d.is_dir() and not d.name.startswith("_temp_")]),
            'total_samples': len(all_smolvlm_data),
            'total_qa_pairs': len(master_df) if 'master_df' in locals() else 0,
            'questions_per_image': self.questions_per_image,
            'delay_between_requests': self.delay_between_requests,
            'max_retries': self.max_retries,
            'timestamp': datetime.now().isoformat()
        }
        
        stats_file = self.output_root / "generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Stats file: {stats_file}")
    
    def process_all_subsets(self, subset_names: List[str] = None, max_images_per_batch: int = 1500):
        """
        Xử lý tất cả subsets, tự động chia nhỏ nếu cần
        Kết quả được gộp về đúng subset gốc
        """
        # Tìm tất cả subsets
        if subset_names is None:
            subsets = [d for d in self.input_root.iterdir() if d.is_dir()]
            # Loại trừ thư mục output và thư mục tạm
            subsets = [d for d in subsets if d.name != self.output_root.name and not d.name.startswith("_temp_split_")]
            subset_names = [d.name for d in subsets]
        
        print(f"\n📦 Tìm thấy {len(subset_names)} subsets: {subset_names}")
        
        all_batches = []
        
        # Kiểm tra và chia nhỏ từng subset
        for subset_name in subset_names:
            image_folder = self.input_root / subset_name
            if not image_folder.exists():
                print(f"⚠️ Không tìm thấy: {image_folder}")
                continue
            
            parts = self._split_subset_if_needed(subset_name, image_folder, max_images_per_batch)
            all_batches.extend(parts)
        
        print(f"\n📦 Tổng số phần cần xử lý: {len(all_batches)}")
        
        if len(all_batches) == 0:
            print("⚠️ Không có phần nào để xử lý!")
            return {}
        
        # Xử lý từng phần
        total_success = 0
        total_images = 0
        total_qa = 0
        
        for part_name, part_folder in all_batches:
            # Lấy tên subset gốc (phần trước _part)
            original_subset = part_name.split("_part")[0]
            
            # Kiểm tra xem phần này đã có kết quả tạm chưa (resume)
            temp_check = self.output_root / f"_temp_{part_name}"
            if self.resume and temp_check.exists():
                print(f"\n🔄 Phần {part_name} đã có kết quả tạm, bỏ qua")
                continue
            
            results = self.process_batch_part(part_name, part_folder, original_subset)
            
            total_success += len([r for r in results if r.get('success')])
            total_images += len(results)
            total_qa += sum([r.get('num_questions', 0) for r in results if r.get('success')])
            
            if not self.key_manager.get_available_key():
                print("\n⚠️ Hết API key! Dừng xử lý")
                break
        
        # Gộp tất cả kết quả vào subset gốc
        self._merge_all_subsets()
        
        # Dọn dẹp thư mục tạm
        self._cleanup_temp_folders()
        
        print(f"\n{'='*60}")
        print(f"📊 TỔNG KẾT")
        print(f"{'='*60}")
        print(f"📸 Tổng số ảnh đã xử lý: {total_images}")
        print(f"✅ Thành công: {total_success}")
        print(f"📝 Tổng QA pairs: {total_qa}")
        print(f"📁 Kết quả lưu tại: {self.output_root}")
        print(f"   - Mỗi subset có thư mục riêng (ví dụ: {subset_names[0] if subset_names else '...'})")
        print(f"   - master_smolvlm_dataset.jsonl (tổng hợp tất cả)")
        
        return all_batches


# ========== SỬ DỤNG ==========

if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa ảnh đã trích xuất
    INPUT_ROOT = r"D:\Study\On-going\Xu_ly_anh\Computer Vision\defect_detection\data\extracted_raw_images"
    OUTPUT_ROOT = r"D:\Study\On-going\Xu_ly_anh\Computer Vision\defect_detection\data\gemini_qa_results"
    
    # Danh sách API keys (thay bằng key thật)
    API_KEYS = [
        "AIzaSyAmZI21KJrppZw5BDZHpZDfwKLCkbYorEg",
        "AIzaSyDQ75-AIzaSyBbTXY2TGtkwRY6V-m8PqLg1C8vdV6GDUw",
        "AIzaSyDFKA_buA9lLhKYk4xVTIQirrPlfC0IiXI",
        "AIzaSyCauBA3n4QrEuIy7MmNdVybuIhB-kT9WJU"
    ]
    
    # Khởi tạo processor
    processor = GeminiQABatchProcessor(
        input_root=Path(INPUT_ROOT),
        api_keys=API_KEYS,
        output_root=Path(OUTPUT_ROOT),
        questions_per_image=5,
        delay_between_requests=10,  # Delay 3 giây giữa các request
        resume=True,
        max_retries=3  # Thử lại tối đa 3 lần khi gặp lỗi
    )
    
    # Xử lý tất cả subsets
    processor.process_all_subsets(
        subset_names=None,  # None = xử lý tất cả
        max_images_per_batch=500  # Tối đa 500 ảnh/batch
    )