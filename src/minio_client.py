"""
MinIO client for image storage
"""
import io
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
from datetime import datetime
import cv2
import numpy as np

from minio import Minio
from minio.error import S3Error

from .config import minio_config


class MinIOClient:
    """
    Client for interacting with MinIO storage
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._client = None
            self._connect()
    
    def _connect(self):
        """Establish connection to MinIO"""
        self._client = Minio(
            endpoint=minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
            region=minio_config.region
        )
        
        # Ensure bucket exists
        self._ensure_bucket()
        
        print(f"✅ Connected to MinIO at {minio_config.endpoint}")
    
    def _ensure_bucket(self):
        """Create bucket if not exists"""
        if not self._client.bucket_exists(minio_config.bucket_name):
            self._client.make_bucket(minio_config.bucket_name)
            print(f"📦 Created bucket: {minio_config.bucket_name}")
    
    def upload_image(
        self,
        image: np.ndarray,
        object_name: str,
        content_type: str = "image/jpeg",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload image to MinIO
        
        Args:
            image: Image as numpy array (BGR format)
            object_name: Object name/path in MinIO
            content_type: MIME type of the image
            metadata: Additional metadata to store with the object
        
        Returns:
            str: Object URL
        """
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = io.BytesIO(buffer)
        image_size = len(buffer)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata['upload_time'] = datetime.now().isoformat()
        
        # Upload to MinIO
        try:
            self._client.put_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            data=image_bytes,
            length=image_size,
            content_type=content_type,
            metadata=metadata
        )
            print(f"✅ Upload thành công: {object_name}") # Thêm dòng này
        except Exception as e:
            print(f"❌ Upload thất bại: {e}") # Thêm dòng này
            raise
        
        # Generate URL
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
        
        return url
    
    def upload_image_file(
        self,
        image_path: Path,
        object_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload image file to MinIO
        
        Args:
            image_path: Path to image file
            object_name: Object name in MinIO (default: use filename)
            metadata: Additional metadata
        
        Returns:
            str: Object URL
        """
        if object_name is None:
            object_name = image_path.name
        
        # Read and upload
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        return self.upload_image(image, object_name, metadata=metadata)
    
    def download_image(self, object_name: str) -> Optional[np.ndarray]:
        """
        Download image from MinIO
        
        Args:
            object_name: Object name/path in MinIO
        
        Returns:
            np.ndarray: Image as numpy array (BGR format)
        """
        try:
            response = self._client.get_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name
            )
            
            # Read image data
            image_data = response.read()
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            response.close()
            response.release_conn()
            
            return image
            
        except S3Error as e:
            print(f"❌ Error downloading {object_name}: {e}")
            return None
    
    def get_object_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get object metadata from MinIO"""
        try:
            response = self._client.stat_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name
            )
            return {
                'size': response.size,
                'etag': response.etag,
                'last_modified': response.last_modified,
                'content_type': response.content_type,
                'metadata': response.metadata
            }
        except S3Error:
            return None
    
    def list_objects(self, prefix: str = "") -> list:
        """List objects in bucket"""
        objects = []
        try:
            for obj in self._client.list_objects(
                bucket_name=minio_config.bucket_name,
                prefix=prefix,
                recursive=True
            ):
                objects.append({
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                })
        except S3Error as e:
            print(f"❌ Error listing objects: {e}")
        
        return objects
    
    def delete_object(self, object_name: str) -> bool:
        """Delete object from MinIO"""
        try:
            self._client.remove_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name
            )
            return True
        except S3Error as e:
            print(f"❌ Error deleting {object_name}: {e}")
            return False
    
    def get_url(self, object_name: str) -> str:
        """Get public URL for object"""
        return f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"