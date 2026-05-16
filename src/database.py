"""
Database models and connection manager for metadata storage
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
import json

from .config import path_config

Base = declarative_base()


# ==================== Database Models ====================

class DefectImage(Base):
    """Lưu thông tin metadata của ảnh"""
    __tablename__ = 'defect_images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String(255), unique=True, nullable=False, index=True)
    subset_name = Column(String(255), nullable=False, index=True)
    label = Column(String(50), index=True)
    
    # Kích thước ảnh
    orig_width = Column(Integer)
    orig_height = Column(Integer)
    
    # Metadata cho feature extraction
    feature_path = Column(String(500))
    feature_scale = Column(Float)
    feature_x_offset = Column(Integer)
    feature_y_offset = Column(Integer)
    
    # Metadata cho segmentation
    segmentation_path = Column(String(500))
    seg_scale = Column(Float)
    seg_x_offset = Column(Integer)
    seg_y_offset = Column(Integer)
    
    # Mask info
    mask_path = Column(String(500))
    has_groundtruth_mask = Column(Boolean, default=False)
    mask_is_auto_generated = Column(Boolean, default=False)
    
    # Thời gian
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    predictions = relationship("PredictionResult", back_populates="image")
    defects = relationship("DefectInstance", back_populates="image", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'image_id': self.image_id,
            'subset_name': self.subset_name,
            'label': self.label,
            'orig_width': self.orig_width,
            'orig_height': self.orig_height,
            'feature_path': self.feature_path,
            'segmentation_path': self.segmentation_path,
            'mask_path': self.mask_path,
            'has_groundtruth_mask': self.has_groundtruth_mask,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DefectInstance(Base):
    """Lưu thông tin từng lỗi riêng biệt trong ảnh"""
    __tablename__ = 'defect_instances'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('defect_images.id'), nullable=False, index=True)
    
    # Thông tin lỗi
    defect_id = Column(Integer)  # Số thứ tự lỗi trong ảnh
    area_pixels = Column(Integer)
    bbox_x = Column(Integer)
    bbox_y = Column(Integer)
    bbox_w = Column(Integer)
    bbox_h = Column(Integer)
    centroid_x = Column(Float)
    centroid_y = Column(Float)
    
    # Phân loại lỗi (nếu có)
    defect_type = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    image = relationship("DefectImage", back_populates="defects")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'defect_id': self.defect_id,
            'area_pixels': self.area_pixels,
            'bbox': [self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h],
            'centroid': (self.centroid_x, self.centroid_y),
            'defect_type': self.defect_type,
            'confidence': self.confidence
        }


class PredictionResult(Base):
    """Lưu kết quả dự đoán của model"""
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('defect_images.id'), nullable=False, index=True)
    
    # Model info
    model_name = Column(String(255))
    model_version = Column(String(50))
    
    # Kết quả
    num_defects = Column(Integer)
    total_area = Column(Integer)
    max_area = Column(Integer)
    min_area = Column(Integer)
    avg_area = Column(Float)
    
    # Inference time
    inference_time_ms = Column(Float)
    
    # Additional info
    additional_info = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    image = relationship("DefectImage", back_populates="predictions")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'model_name': self.model_name,
            'num_defects': self.num_defects,
            'total_area': self.total_area,
            'max_area': self.max_area,
            'min_area': self.min_area,
            'avg_area': self.avg_area,
            'inference_time_ms': self.inference_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class VQAHistory(Base):
    """Lưu lịch sử câu hỏi và câu trả lời VQA"""
    __tablename__ = 'vqa_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('defect_images.id'), nullable=False, index=True)
    
    # Question and answer
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    question_language = Column(String(10), default='en')
    
    # Model info
    model_name = Column(String(255))
    model_version = Column(String(50))
    
    # Confidence
    confidence = Column(Float, nullable=True)
    
    # Response time
    response_time_ms = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'model_name': self.model_name,
            'confidence': self.confidence,
            'response_time_ms': self.response_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TrainingLog(Base):
    """Lưu log quá trình training"""
    __tablename__ = 'training_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False, index=True)  # 'segmentation', 'vqa'
    model_name = Column(String(255))
    model_version = Column(String(50))
    
    # Training params
    num_epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    
    # Results
    best_val_loss = Column(Float)
    best_val_metric = Column(Float)  # mIoU for segmentation, Accuracy for VQA
    
    # Paths
    model_path = Column(String(500))
    checkpoint_path = Column(String(500))
    
    # Status
    status = Column(String(20), default='completed')  # 'running', 'completed', 'failed'
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'model_path': self.model_path,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# ==================== Database Manager ====================

class DatabaseManager:
    """Quản lý kết nối và thao tác database"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._engine = None
            self._session_factory = None
            self._config = self._load_db_config()
    
    def _load_db_config(self) -> Dict[str, Any]:
        """Load database configuration"""
        # Có thể đọc từ file config hoặc environment variables
        return {
            'db_type': os.environ.get('DB_TYPE', 'sqlite'),  # 'sqlite', 'postgresql', 'mysql'
            'db_path': os.environ.get('DB_PATH', os.path.join(path_config.project_root, 'metadata.db')),
            'host': os.environ.get('DB_HOST', 'localhost'),
            'port': os.environ.get('DB_PORT', '5432'),
            'database': os.environ.get('DB_NAME', 'defect_detection'),
            'username': os.environ.get('DB_USER', ''),
            'password': os.environ.get('DB_PASS', '')
        }
    
    def _get_connection_string(self) -> str:
        """Get database connection string"""
        config = self._config
        
        if config['db_type'] == 'sqlite':
            return f"sqlite:///{config['db_path']}"
        
        elif config['db_type'] == 'postgresql':
            return f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        
        elif config['db_type'] == 'mysql':
            return f"mysql+pymysql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        
        else:
            raise ValueError(f"Unsupported database type: {config['db_type']}")
    
    def connect(self):
        """Establish database connection"""
        if self._engine is None:
            conn_string = self._get_connection_string()
            
            # Create engine with connection pool for production
            if self._config['db_type'] != 'sqlite':
                self._engine = create_engine(
                    conn_string,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True
                )
            else:
                self._engine = create_engine(conn_string)
            
            self._session_factory = sessionmaker(bind=self._engine)
            
            # Create tables if not exists
            Base.metadata.create_all(self._engine)
            print(f"Connected to database: {self._config['db_type']}")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if self._session_factory is None:
            self.connect()
        return self._session_factory()
    
    def close(self):
        """Close database connection"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


# ==================== Data Access Layer ====================

class ImageRepository:
    """Repository for DefectImage operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def add(self, image_data: Dict[str, Any]) -> DefectImage:
        """Add a new image record and return the object (session closed)"""
        session = self.db.get_session()
        try:
            image = DefectImage(**image_data)
            session.add(image)
            session.commit()
            # ✅ Detach object để có thể dùng sau khi session đóng
            session.expunge(image)
            return image
        finally:
            session.close()
    
    def add_and_get_id(self, image_data: Dict[str, Any]) -> int:
        """Add a new image record and return its ID only"""
        session = self.db.get_session()
        try:
            image = DefectImage(**image_data)
            session.add(image)
            session.commit()
            image_id = image.id  # Lấy ID trước khi đóng session
            return image_id
        finally:
            session.close()
    
    def add_defect_instance(self, image_id: int, defect_data: Dict[str, Any]) -> Optional[int]:
        """Add a defect instance and return its ID"""
        session = self.db.get_session()
        try:
            defect = DefectInstance(image_id=image_id, **defect_data)
            session.add(defect)
            session.commit()
            return defect.id
        except Exception as e:
            session.rollback()
            print(f"⚠️ Failed to add defect instance: {e}")
            return None
        finally:
            session.close()
    
    def get_by_id(self, image_id: str) -> Optional[DefectImage]:
        """Get image by image_id"""
        session = self.db.get_session()
        try:
            return session.query(DefectImage).filter(DefectImage.image_id == image_id).first()
        finally:
            session.close()
    
    def get_by_db_id(self, db_id: int) -> Optional[DefectImage]:
        """Get image by database ID"""
        session = self.db.get_session()
        try:
            return session.query(DefectImage).filter(DefectImage.id == db_id).first()
        finally:
            session.close()
    
    def get_by_subset(self, subset_name: str, limit: int = 100) -> List[DefectImage]:
        """Get images by subset name"""
        session = self.db.get_session()
        try:
            return session.query(DefectImage).filter(DefectImage.subset_name == subset_name).limit(limit).all()
        finally:
            session.close()
    
    def update(self, image_id: str, updates: Dict[str, Any]) -> Optional[DefectImage]:
        """Update image record"""
        session = self.db.get_session()
        try:
            image = session.query(DefectImage).filter(DefectImage.image_id == image_id).first()
            if image:
                for key, value in updates.items():
                    if hasattr(image, key):
                        setattr(image, key, value)
                session.commit()
                session.expunge(image)
            return image
        finally:
            session.close()
    
    def get_defects_by_image(self, image_id: int) -> List[Dict[str, Any]]:
        """Get all defect instances for an image as dicts (safe to use after session closes)"""
        session = self.db.get_session()
        try:
            defects = session.query(DefectInstance).filter(DefectInstance.image_id == image_id).all()
            # Convert to dicts before closing session
            return [defect.to_dict() for defect in defects]
        finally:
            session.close()


class PredictionRepository:
    """Repository for PredictionResult operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def add_and_get_id(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """Save prediction result and return ID"""
        session = self.db.get_session()
        try:
            prediction = PredictionResult(**prediction_data)
            session.add(prediction)
            session.commit()
            return prediction.id
        except Exception as e:
            session.rollback()
            print(f"⚠️ Failed to add prediction: {e}")
            return None
        finally:
            session.close()
    
    def add(self, prediction_data: Dict[str, Any]) -> PredictionResult:
        """Save prediction result"""
        session = self.db.get_session()
        try:
            prediction = PredictionResult(**prediction_data)
            session.add(prediction)
            session.commit()
            return prediction
        finally:
            session.close()
    
    def get_by_image(self, image_id: int, limit: int = 10) -> List[PredictionResult]:
        """Get prediction history for an image"""
        session = self.db.get_session()
        try:
            return session.query(PredictionResult).filter(PredictionResult.image_id == image_id)\
                .order_by(PredictionResult.created_at.desc()).limit(limit).all()
        finally:
            session.close()


class VQARepository:
    """Repository for VQA operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def add(self, vqa_data: Dict[str, Any]) -> VQAHistory:
        """Save Q&A history"""
        session = self.db.get_session()
        try:
            vqa_log = VQAHistory(**vqa_data)
            session.add(vqa_log)
            session.commit()
            return vqa_log
        finally:
            session.close()
    
    def get_history_by_image(self, image_id: int, limit: int = 50) -> List[VQAHistory]:
        """Get Q&A history for an image"""
        session = self.db.get_session()
        try:
            return session.query(VQAHistory).filter(VQAHistory.image_id == image_id)\
                .order_by(VQAHistory.created_at.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_recent_questions(self, limit: int = 100) -> List[VQAHistory]:
        """Get most recent questions"""
        session = self.db.get_session()
        try:
            return session.query(VQAHistory).order_by(VQAHistory.created_at.desc()).limit(limit).all()
        finally:
            session.close()
    
    def add_and_get_id(self, vqa_data: Dict[str, Any]) -> Optional[int]:
        """Save Q&A history and return ID"""
        session = self.db.get_session()
        try:
            vqa_log = VQAHistory(**vqa_data)
            session.add(vqa_log)
            session.commit()
            return vqa_log.id
        except Exception as e:
            session.rollback()
            print(f"⚠️ Failed to add VQA history: {e}")
            return None
        finally:
            session.close()


class TrainingRepository:
    """Repository for TrainingLog operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def add(self, training_data: Dict[str, Any]) -> TrainingLog:
        """Save training log"""
        session = self.db.get_session()
        try:
            log = TrainingLog(**training_data)
            session.add(log)
            session.commit()
            return log
        finally:
            session.close()
    
    def update_status(self, log_id: int, status: str, error_message: str = None):
        """Update training status"""
        session = self.db.get_session()
        try:
            log = session.query(TrainingLog).filter(TrainingLog.id == log_id).first()
            if log:
                log.status = status
                if error_message:
                    log.error_message = error_message
                if status == 'completed':
                    log.completed_at = datetime.now()
                session.commit()
        finally:
            session.close()
    
    def get_best_model(self, model_type: str) -> Optional[TrainingLog]:
        """Get best model of a specific type"""
        session = self.db.get_session()
        try:
            return session.query(TrainingLog).filter(
                TrainingLog.model_type == model_type,
                TrainingLog.status == 'completed'
            ).order_by(TrainingLog.best_val_metric.desc()).first()
        finally:
            session.close()