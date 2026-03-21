import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import FileValidationError, StorageError

logger = logging.getLogger(__name__)

MIME_TYPES = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "application/pdf": "pdf",
}


def validate_file(file: UploadFile) -> str:
    content_type = file.content_type or ""
    ext = MIME_TYPES.get(content_type.lower())

    if not ext:
        if file.filename:
            name_ext = Path(file.filename).suffix.lower().lstrip(".")
            if name_ext in settings.ALLOWED_EXTENSIONS:
                ext = name_ext
            else:
                raise FileValidationError(
                    f"Unsupported file type: {content_type}. "
                    f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                )
        else:
            raise FileValidationError("Cannot determine file type")

    return ext


async def save_upload(file: UploadFile) -> Tuple[str, str, int, str]:
    ext = validate_file(file)

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    unique_id = uuid.uuid4().hex
    safe_name = f"{unique_id}.{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    content = await file.read()
    file_size = len(content)

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise FileValidationError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB. "
            f"Maximum: {settings.MAX_FILE_SIZE_MB}MB"
        )

    if file_size == 0:
        raise FileValidationError("File is empty")

    try:
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved upload: {safe_name} ({file_size} bytes)")
    except OSError as e:
        raise StorageError(f"Failed to save file: {str(e)}")

    original_name = file.filename or safe_name
    return file_path, original_name, file_size, file.content_type or f"image/{ext}"


def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            base, ext = os.path.splitext(file_path)
            preprocessed = f"{base}_preprocessed{ext}"
            if os.path.exists(preprocessed):
                os.remove(preprocessed)
    except OSError as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")