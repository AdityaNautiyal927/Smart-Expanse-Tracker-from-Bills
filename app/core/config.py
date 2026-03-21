from pydantic_settings import BaseSettings
from typing import List
import os
 
 
class Settings(BaseSettings):
    APP_NAME: str = "Smart Expense Tracker"
    DEBUG: bool = True
    VERSION: str = "1.0.0"
 
    DATABASE_URL: str = "sqlite+aiosqlite:///./expenses.db"
 
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "pdf"]
 
    OCR_ENGINE: str = "easyocr"  
    OCR_LANGUAGES: List[str] = ["en"]
    OCR_GPU: bool = False  
 
    ENABLE_ASYNC_PROCESSING: bool = False  # True → uses Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
 
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600
 
    USE_S3: bool = False
    AWS_BUCKET_NAME: str = ""
    AWS_REGION: str = "us-east-1"
 
    ALLOWED_ORIGINS: List[str] = ["*"]

    CATEGORIZATION_ENGINE: str = "rule_based"  # "rule_based" | "ml_model" | "hybrid"
    ML_MODEL_PATH: str = "./models/categorizer.pkl"
 
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
 
 
settings = Settings()