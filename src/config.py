"""
Configuration management for the defect detection project
"""
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import yaml


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str = field(default_factory=lambda: os.environ.get('DB_TYPE', 'sqlite'))
    db_path: str = field(default_factory=lambda: os.environ.get('DB_PATH', './metadata.db'))
    host: str = field(default_factory=lambda: os.environ.get('DB_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.environ.get('DB_PORT', '5432')))
    database: str = field(default_factory=lambda: os.environ.get('DB_NAME', 'defect_detection'))
    username: str = field(default_factory=lambda: os.environ.get('DB_USER', ''))
    password: str = field(default_factory=lambda: os.environ.get('DB_PASS', ''))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseConfig':
        return cls(
            db_type=data.get('type', 'sqlite'),
            db_path=data.get('path', './metadata.db'),
            host=data.get('host', 'localhost'),
            port=data.get('port', 5432),
            database=data.get('name', 'defect_detection'),
            username=data.get('user', ''),
            password=data.get('password', '')
        )
    
    def get_connection_string(self) -> str:
        if self.db_type == 'sqlite':
            return f"sqlite:///{self.db_path}"
        elif self.db_type == 'postgresql':
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == 'mysql':
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class MinIOConfig:
    """MinIO configuration for image storage"""
    endpoint: str = field(default_factory=lambda: os.environ.get('MINIO_ENDPOINT', 'localhost:9000'))
    access_key: str = field(default_factory=lambda: os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'))
    secret_key: str = field(default_factory=lambda: os.environ.get('MINIO_SECRET_KEY', 'minioadmin'))
    bucket_name: str = field(default_factory=lambda: os.environ.get('MINIO_BUCKET', 'defect-images'))
    secure: bool = False
    region: str = "us-east-1"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MinIOConfig':
        return cls(
            endpoint=data.get('endpoint', os.environ.get('MINIO_ENDPOINT', 'localhost:9000')),
            access_key=data.get('access_key', os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')),
            secret_key=data.get('secret_key', os.environ.get('MINIO_SECRET_KEY', 'minioadmin')),
            bucket_name=data.get('bucket_name', os.environ.get('MINIO_BUCKET', 'defect-images')),
            secure=data.get('secure', False),
            region=data.get('region', 'us-east-1')
        )


@dataclass
class PathConfig:
    """Đường dẫn các thư mục trong project"""
    project_root: str = None
    
    def __post_init__(self):
        if self.project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    _data_raw: str = None
    _data_preprocessed: str = None
    _logs_dir: str = None
    _models_dir: str = None
    _segmentation_models: str = None
    _vqa_models: str = None
    _checkpoints: str = None
    _config_file: str = None
    
    @property
    def data_raw(self) -> str:
        if self._data_raw is not None:
            return self._data_raw
        return os.path.join(self.project_root, 'data', 'raw')
    
    @data_raw.setter
    def data_raw(self, value: str):
        self._data_raw = value
    
    @property
    def data_preprocessed(self) -> str:
        if self._data_preprocessed is not None:
            return self._data_preprocessed
        return os.path.join(self.project_root, 'data', 'preprocessed')
    
    @data_preprocessed.setter
    def data_preprocessed(self, value: str):
        self._data_preprocessed = value
    
    @property
    def logs_dir(self) -> str:
        if self._logs_dir is not None:
            return self._logs_dir
        return os.path.join(self.project_root, 'logs')
    
    @logs_dir.setter
    def logs_dir(self, value: str):
        self._logs_dir = value
    
    @property
    def models_dir(self) -> str:
        if self._models_dir is not None:
            return self._models_dir
        return os.path.join(self.project_root, 'models')
    
    @models_dir.setter
    def models_dir(self, value: str):
        self._models_dir = value
    
    @property
    def segmentation_models(self) -> str:
        if self._segmentation_models is not None:
            return self._segmentation_models
        return os.path.join(self.models_dir, 'segmentation')
    
    @segmentation_models.setter
    def segmentation_models(self, value: str):
        self._segmentation_models = value
    
    @property
    def vqa_models(self) -> str:
        if self._vqa_models is not None:
            return self._vqa_models
        return os.path.join(self.models_dir, 'vqa')
    
    @vqa_models.setter
    def vqa_models(self, value: str):
        self._vqa_models = value
    
    @property
    def checkpoints(self) -> str:
        if self._checkpoints is not None:
            return self._checkpoints
        return os.path.join(self.models_dir, 'checkpoints')
    
    @checkpoints.setter
    def checkpoints(self, value: str):
        self._checkpoints = value
    
    @property
    def config_file(self) -> str:
        if self._config_file is not None:
            return self._config_file
        return os.path.join(self.project_root, 'config.yaml')
    
    @config_file.setter
    def config_file(self, value: str):
        self._config_file = value
    
    def update_from_dict(self, data: Dict[str, str]):
        for key, value in data.items():
            if hasattr(self, key) and isinstance(getattr(type(self), key, None), property):
                setattr(self, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)


@dataclass
class PreprocessingConfig:
    """Cấu hình tiền xử lý"""
    median_kernel: int = 5
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    use_padding: bool = True
    
    # Resize sizes
    resize_for_feature: Tuple[int, int] = (224, 224)
    resize_for_segmentation: Tuple[int, int] = (512, 512)
    
    # Normalization (grayscale)
    mean: float = 0.456
    std: float = 0.224


@dataclass
class SegmentationConfig:
    """Cấu hình cho U-Net segmentation"""
    encoder_name: str = 'efficientnet-b0'
    in_channels: int = 1
    num_classes: int = 2
    
    # Training
    batch_size: int = 8
    input_size: Tuple[int, int] = (512, 512)
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    threshold: float = 0.5
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.3
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = 'min'
    
    def update_from_dict(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self, key):
                if key in ['input_size'] and isinstance(value, list):
                    setattr(self, key, tuple(value))
                else:
                    setattr(self, key, value)


@dataclass
class VQAConfig:
    """Cấu hình cho VQA model"""
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct"
    device: str = "cpu"
    max_new_tokens: int = 150
    temperature: float = 0.1
    
    # Question preprocessing
    max_question_len: int = 64
    bert_model: str = 'bert-base-uncased'
    freeze_bert: bool = True
    
    # Model
    feature_dim: int = 512
    fusion_dim: int = 512
    num_attention_heads: int = 8
    
    # Training
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Validation
    val_split: float = 0.2
    early_stopping_patience: int = 5


def load_config_from_yaml() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config.yaml: {e}")
    return {}


def update_config_from_file():
    yaml_config = load_config_from_yaml()
    
    if 'database' in yaml_config:
        global db_config
        db_config = DatabaseConfig.from_dict(yaml_config['database'])
    
    if 'minio' in yaml_config:
        global minio_config
        minio_config = MinIOConfig.from_dict(yaml_config['minio'])
    
    if 'paths' in yaml_config:
        path_config.update_from_dict(yaml_config['paths'])
    
    if 'segmentation' in yaml_config:
        seg_config.update_from_dict(yaml_config['segmentation'])
    
    if 'vqa' in yaml_config:
        vqa = yaml_config['vqa']
        if 'model_name' in vqa:
            vqa_config.model_name = vqa['model_name']
        if 'device' in vqa:
            vqa_config.device = vqa['device']
        if 'max_new_tokens' in vqa:
            vqa_config.max_new_tokens = vqa['max_new_tokens']
        if 'temperature' in vqa:
            vqa_config.temperature = vqa['temperature']
        if 'max_question_len' in vqa:
            vqa_config.max_question_len = vqa['max_question_len']
        if 'bert_model' in vqa:
            vqa_config.bert_model = vqa['bert_model']
        if 'batch_size' in vqa:
            vqa_config.batch_size = vqa['batch_size']
        if 'num_epochs' in vqa:
            vqa_config.num_epochs = vqa['num_epochs']
        if 'learning_rate' in vqa:
            vqa_config.learning_rate = vqa['learning_rate']
    
    if 'preprocessing' in yaml_config:
        preproc = yaml_config['preprocessing']
        for key, value in preproc.items():
            if hasattr(preprocess_config, key):
                if key in ['resize_for_feature', 'resize_for_segmentation', 'clahe_grid_size'] and isinstance(value, list):
                    setattr(preprocess_config, key, tuple(value))
                else:
                    setattr(preprocess_config, key, value)


# Singleton instances
path_config = PathConfig()
db_config = DatabaseConfig()
minio_config = MinIOConfig()
preprocess_config = PreprocessingConfig()
seg_config = SegmentationConfig()
vqa_config = VQAConfig()

# Load from YAML if exists
update_config_from_file()


def get_config_summary() -> Dict[str, Any]:
    return {
        'database': {
            'type': db_config.db_type,
            'path': db_config.db_path
        },
        'minio': {
            'endpoint': minio_config.endpoint,
            'bucket': minio_config.bucket_name
        },
        'paths': {
            'project_root': path_config.project_root,
            'data_raw': path_config.data_raw,
            'data_preprocessed': path_config.data_preprocessed,
            'logs_dir': path_config.logs_dir,
            'models_dir': path_config.models_dir
        },
        'segmentation': {
            'encoder_name': seg_config.encoder_name,
            'input_size': seg_config.input_size,
            'batch_size': seg_config.batch_size,
            'num_epochs': seg_config.num_epochs,
            'learning_rate': seg_config.learning_rate,
            'early_stopping_patience': seg_config.early_stopping_patience
        },
        'preprocessing': {
            'median_kernel': preprocess_config.median_kernel,
            'resize_for_feature': preprocess_config.resize_for_feature,
            'resize_for_segmentation': preprocess_config.resize_for_segmentation
        }
    }


if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Project root: {path_config.project_root}")
    print(f"Data raw: {path_config.data_raw}")
    print(f"Data preprocessed: {path_config.data_preprocessed}")
    print(f"Segmentation model path: {path_config.segmentation_models}")