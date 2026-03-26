from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "ocr_engine": settings.OCR_ENGINE,
        "categorization_engine": settings.CATEGORIZATION_ENGINE,
        "async_processing": settings.ENABLE_ASYNC_PROCESSING,
    }