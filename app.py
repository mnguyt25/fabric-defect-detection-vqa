"""
FastAPI web application for defect detection and VQA system
"""
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np

from src.database import DatabaseManager, ImageRepository, PredictionRepository, VQARepository
from src.minio_client import MinIOClient
from src.segmentation import SegmentationModel

# Tạo thư mục tạm thời
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Khởi tạo FastAPI
app = FastAPI(title="Defect Detection System", version="1.0.0")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Khởi tạo các service
_db_manager = None
_image_repo = None
_prediction_repo = None
_vqa_repo = None
_minio_client = None
_segmentation_model = None
_vlm_engine = None


def init_services():
    """Khởi tạo các service (lazy loading)"""
    global _db_manager, _image_repo, _prediction_repo, _vqa_repo, _minio_client, _segmentation_model, _vlm_engine
    
    if _db_manager is None:
        print("🚀 Initializing services...")
        
        # Database
        _db_manager = DatabaseManager()
        _db_manager.connect()
        _image_repo = ImageRepository(_db_manager)
        _prediction_repo = PredictionRepository(_db_manager)
        _vqa_repo = VQARepository(_db_manager)
        
        # MinIO (optional)
        try:
            _minio_client = MinIOClient()
            print("✅ MinIO client initialized")
        except Exception as e:
            print(f"⚠️ MinIO not available: {e}")
            _minio_client = None
        
        # Segmentation Model
        try:
            _segmentation_model = SegmentationModel(device="cpu")
        except Exception as e:
            print(f"⚠️ Segmentation model not loaded: {e}")
            _segmentation_model = None
        
        # VLM Engine (SmolVLM)
        try:
            from src.vqa import SmolVLMEngine
            _vlm_engine = SmolVLMEngine(device="cpu", segmentation_model=_segmentation_model)
            print("✅ SmolVLM loaded successfully!")
        except Exception as e:
            print(f"⚠️ SmolVLM not loaded: {e}")
            _vlm_engine = None
        
        print("✅ Services initialized!")


def get_segmentation_model():
    init_services()
    return _segmentation_model


def get_vlm_engine():
    init_services()
    return _vlm_engine


def get_image_repo():
    init_services()
    return _image_repo


def get_minio_client():
    init_services()
    return _minio_client


def generate_caption(num_defects: int) -> str:
    """Tạo caption đơn giản"""
    if num_defects == 0:
        return "✅ No defects detected on this fabric."
    elif num_defects == 1:
        return f"⚠️ Detected {num_defects} defect on the fabric."
    else:
        return f"⚠️ Detected {num_defects} defects on the fabric."


def visualize_result(image: np.ndarray, mask: np.ndarray, defects: list) -> np.ndarray:
    """Vẽ kết quả lên ảnh"""
    overlay = image.copy()
    
    # Vẽ mask màu đỏ
    overlay[mask == 1] = [0, 0, 255]
    
    # Blend với ảnh gốc
    vis_image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    # Vẽ bounding boxes
    for defect in defects:
        x, y, w, h = defect['bbox']
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis_image, f"#{defect['id']+1}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_image


def upload_to_minio(image: np.ndarray, image_id: str, subset_name: str = "web_upload") -> Optional[str]:
    """
    Upload ảnh lên MinIO sử dụng MinIOClient
    
    Args:
        image: Ảnh dạng numpy array (BGR format)
        image_id: ID duy nhất của ảnh
        subset_name: Tên subset (folder ảo trên MinIO)
    
    Returns:
        URL của ảnh trên MinIO hoặc None nếu upload thất bại
    """
    minio_client = get_minio_client()
    
    if minio_client is None:
        print("⚠️ MinIO client not available, skipping upload")
        return None
    
    try:
        # Tạo object_name theo format: subset_name/image_id.jpg
        object_name = f"{subset_name}/{image_id}.jpg"
        
        # Tạo metadata cho ảnh
        metadata = {
            'upload_time': datetime.now().isoformat(),
            'source': 'web_upload',
            'image_id': image_id,
            'subset_name': subset_name
        }
        
        # Upload ảnh lên MinIO
        # Lưu ý: upload_image trả về URL của ảnh
        url = minio_client.upload_image(
            image=image,
            object_name=object_name,
            content_type="image/jpeg",
            metadata=metadata
        )
        
        print(f"✅ Image uploaded to MinIO: {object_name}")
        print(f"📍 URL: {url}")
        return url
        
    except Exception as e:
        print(f"❌ Failed to upload to MinIO: {e}")
        return None


def save_to_database(
    image_id: str,
    subset_name: str,
    label: str,
    image_shape: tuple,
    minio_url: Optional[str],
    seg_result: dict
) -> Optional[int]:
    """
    Lưu thông tin vào database
    
    Returns:
        image_record_id hoặc None
    """
    try:
        repo = get_image_repo()
        if repo is None:
            print("⚠️ Image repository not available")
            return None
        
        # ✅ Dùng add_and_get_id để chỉ lấy ID, không giữ object
        record_id = repo.add_and_get_id({
            'image_id': image_id,
            'subset_name': subset_name,
            'label': label,
            'orig_width': image_shape[1],
            'orig_height': image_shape[0],
            'feature_path': minio_url or "",
            'segmentation_path': minio_url or "",
            'has_groundtruth_mask': False,
            'mask_is_auto_generated': True
        })
        
        if record_id is None:
            print("⚠️ Failed to get record ID")
            return None
        
        # ✅ Lưu từng defect instance
        defects_added = 0
        for defect in seg_result.get('defects', []):
            result = repo.add_defect_instance(record_id, {
                'defect_id': defect['id'],
                'area_pixels': defect['area'],
                'bbox_x': defect['bbox'][0],
                'bbox_y': defect['bbox'][1],
                'bbox_w': defect['bbox'][2],
                'bbox_h': defect['bbox'][3],
                'centroid_x': defect['centroid'][0],
                'centroid_y': defect['centroid'][1],
                'defect_type': None,
                'confidence': None
            })
            if result is not None:
                defects_added += 1
        
        print(f"✅ Saved to database: {image_id} (ID: {record_id}, Defects: {defects_added}/{len(seg_result.get('defects', []))})")
        return record_id
        
    except Exception as e:
        print(f"⚠️ Failed to save to database: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect_defects(
    request: Request,
    file: UploadFile = File(...),
    save_to_minio: bool = Form(True),
    save_to_db: bool = Form(True)
):
    """Upload và phân tích ảnh - phát hiện lỗi và lưu lên MinIO + Database"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # Tạo ID duy nhất cho ảnh
    image_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    temp_filename = f"{image_id}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_filename
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        image = cv2.imread(str(temp_path))
        if image is None:
            raise HTTPException(400, "Cannot read image file")
        
        model = get_segmentation_model()
        minio_url = None
        
        if model is None:
            # Fallback demo
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            defects = []
            num_defects = 0
            caption = generate_caption(num_defects)
            vis_image = visualize_result(image, mask, defects)
            vis_path = UPLOAD_DIR / f"{temp_filename}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_image)
            
            # Upload lên MinIO nếu được yêu cầu
            if save_to_minio:
                minio_url = upload_to_minio(image, image_id, "web_demo")
            
            # Lưu vào database nếu được yêu cầu
            if save_to_db:
                save_to_database(
                    image_id=image_id,
                    subset_name="web_demo",
                    label="demo",
                    image_shape=image.shape,
                    minio_url=minio_url,
                    seg_result={'num_defects': 0, 'defects': []}
                )
            
            return JSONResponse({
                'success': True,
                'image_id': image_id,
                'minio_url': minio_url,
                'num_defects': num_defects,
                'caption': caption,
                'visualization_url': f"/uploads/{temp_filename}_vis.jpg",
                'defects': defects,
                'segmentation_time': 0,
                'warning': 'Segmentation model not available. Using demo mode.'
            })
        
        # Chạy segmentation
        seg_result = model.predict(image)
        caption = generate_caption(seg_result['num_defects'])
        vis_image = visualize_result(image, seg_result['mask'], seg_result['defects'])
        vis_path = UPLOAD_DIR / f"{temp_filename}_vis.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        
        # Upload lên MinIO nếu được yêu cầu
        if save_to_minio:
            minio_url = upload_to_minio(image, image_id, "web_detection")
        
        # Lưu vào database nếu được yêu cầu
        if save_to_db:
            save_to_database(
                image_id=image_id,
                subset_name="web_detection",
                label="detected",
                image_shape=image.shape,
                minio_url=minio_url,
                seg_result=seg_result
            )
        
        return JSONResponse({
            'success': True,
            'image_id': image_id,
            'minio_url': minio_url,
            'num_defects': seg_result['num_defects'],
            'caption': caption,
            'visualization_url': f"/uploads/{temp_filename}_vis.jpg",
            'defects': seg_result['defects'],
            'segmentation_time': 0,
            'saved_to_minio': minio_url is not None,
            'saved_to_db': save_to_db
        })
        
    except Exception as e:
        raise HTTPException(500, str(e))
    
    finally:
        # Xóa file tạm (giữ lại file visualization)
        if temp_path.exists() and "_vis" not in temp_path.name:
            temp_path.unlink()


@app.post("/vqa")
async def visual_question_answering(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    save_to_minio: bool = Form(True),
    save_to_db: bool = Form(True)
):
    """Visual Question Answering - hỏi đáp về ảnh"""
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    vlm = get_vlm_engine()
    if vlm is None:
        return JSONResponse({
            'success': False,
            'error': 'VQA model not available. Please check SmolVLM installation.'
        })
    
    # Tạo ID duy nhất cho ảnh
    image_id = f"vqa_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    temp_filename = f"{image_id}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_filename
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        image = cv2.imread(str(temp_path))
        if image is None:
            raise HTTPException(400, "Cannot read image file")
        
        # Chạy segmentation để lấy context
        model = get_segmentation_model()
        seg_result = None
        
        if model:
            seg_result = model.predict(image)
            num_defects = seg_result['num_defects']
            # Sử dụng VLM với segmentation context (đã tích hợp trong vlm.answer)
        
        # Trả lời câu hỏi bằng SmolVLM (đã tích hợp UNet context bên trong)
        answer = vlm.answer(
            image=image,
            question=question,
            use_segmentation=True,
            use_rule_based=True
        )
        
        # Upload lên MinIO nếu được yêu cầu
        minio_url = None
        if save_to_minio:
            minio_url = upload_to_minio(image, image_id, "vqa_images")
        
        # Lưu vào database nếu được yêu cầu
        if save_to_db and seg_result:
            save_to_database(
                image_id=image_id,
                subset_name="vqa_images",
                label="vqa_queried",
                image_shape=image.shape,
                minio_url=minio_url,
                seg_result=seg_result
            )
        
        return JSONResponse({
            'success': True,
            'image_id': image_id,
            'minio_url': minio_url,
            'answer': answer,
            'question': question,
            'num_defects': seg_result['num_defects'] if seg_result else None,
            'saved_to_minio': minio_url is not None,
            'saved_to_db': save_to_db
        })
        
    except Exception as e:
        raise HTTPException(500, str(e))
    
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Lấy file đã upload/xử lý"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)


@app.get("/images/minio/{object_name:path}")
async def get_image_from_minio(object_name: str):
    """
    Lấy ảnh trực tiếp từ MinIO
    
    Args:
        object_name: Đường dẫn object trong MinIO (ví dụ: web_detection/web_20240101_xxx.jpg)
    """
    minio_client = get_minio_client()
    
    if minio_client is None:
        raise HTTPException(503, "MinIO service not available")
    
    try:
        # Download ảnh từ MinIO
        image = minio_client.download_image(object_name)
        if image is None:
            raise HTTPException(404, f"Image not found in MinIO: {object_name}")
        
        # Chuyển đổi sang bytes để gửi về
        _, buffer = cv2.imencode('.jpg', image)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(500, f"Error fetching image: {e}")


@app.get("/minio/objects")
async def list_minio_objects(prefix: str = ""):
    """Liệt kê các object trong MinIO bucket"""
    minio_client = get_minio_client()
    
    if minio_client is None:
        raise HTTPException(503, "MinIO service not available")
    
    try:
        objects = minio_client.list_objects(prefix=prefix)
        return JSONResponse({
            'success': True,
            'objects': objects,
            'count': len(objects)
        })
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': str(e)
        })


@app.get("/history")
async def get_history(limit: int = 50):
    """Lấy lịch sử các ảnh đã xử lý từ database"""
    
    if _db_manager is None:
        return JSONResponse({
            'success': False,
            'error': 'Database not initialized'
        })
    
    try:
        from src.database import DefectImage
        
        session = _db_manager.get_session()
        try:
            images = session.query(DefectImage).order_by(
                DefectImage.created_at.desc()
            ).limit(limit).all()
            
            # ✅ Chuyển đổi dữ liệu trước khi đóng session
            result = []
            for img in images:
                img_dict = {
                    'id': img.id,
                    'image_id': img.image_id,
                    'subset_name': img.subset_name,
                    'label': img.label,
                    'orig_width': img.orig_width,
                    'orig_height': img.orig_height,
                    'feature_path': img.feature_path,
                    'segmentation_path': img.segmentation_path,
                    'mask_path': img.mask_path,
                    'has_groundtruth_mask': img.has_groundtruth_mask,
                    'created_at': img.created_at.isoformat() if img.created_at else None
                }
                
                # Thêm URL MinIO nếu có
                if img_dict.get('feature_path') and 'minio' in img_dict['feature_path']:
                    parts = img_dict['feature_path'].split('/')
                    if len(parts) >= 2:
                        img_dict['minio_object'] = '/'.join(parts[-2:])
                result.append(img_dict)
            
            return JSONResponse({
                'success': True,
                'images': result,
                'count': len(result)
            })
        finally:
            session.close()
        
    except Exception as e:
        print(f"Error in /history: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            'success': False,
            'error': str(e)
        })


@app.get("/health")
async def health_check():
    """Kiểm tra sức khỏe hệ thống"""
    minio_client = get_minio_client()
    minio_status = minio_client is not None
    
    # Kiểm tra kết nối MinIO thực tế
    if minio_status:
        try:
            # Thử list objects để kiểm tra kết nối
            minio_client.list_objects(prefix="", max_keys=1)
        except:
            minio_status = False
    
    db_status = get_image_repo() is not None
    seg_status = get_segmentation_model() is not None
    vlm_status = get_vlm_engine() is not None
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "minio": minio_status,
            "database": db_status,
            "segmentation": seg_status,
            "vlm": vlm_status
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("🚀 Starting Defect Detection & VQA Web System")
    print("="*60)
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"🌐 Web interface: http://localhost:8000")
    print(f"📦 MinIO endpoint: http://localhost:9000")
    print("="*60)
    uvicorn.run(app, host="127.0.0.1", port=8000)