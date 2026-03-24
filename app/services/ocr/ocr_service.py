import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
 
from app.core.config import settings
from app.core.exceptions import OCRError
from app.schemas.expense import OCRResult
 
logger = logging.getLogger(__name__)
 

 
class BaseOCREngine(ABC):
    """Contract all OCR engines must fulfill."""
 
    @abstractmethod
    def extract_text(self, image_path: str) -> tuple[str, float]:
        """
        Extract text from image.
        Returns: (raw_text, confidence_score 0-1)
        """
        ...
 
    @property
    @abstractmethod
    def engine_name(self) -> str:
        ...
 
 
class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR-based text extraction.
    Supports 80+ languages, no internet required.
    """
 
    def __init__(self):
        self._reader = None  
 
    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR (GPU={settings.OCR_GPU})...")
                self._reader = easyocr.Reader(
                    settings.OCR_LANGUAGES,
                    gpu=settings.OCR_GPU
                )
                logger.info("EasyOCR initialized ✅")
            except ImportError:
                raise OCRError(
                    "EasyOCR not installed. Run: pip install easyocr"
                )
        return self._reader
 
    def extract_text(self, image_path: str) -> tuple[str, float]:
        reader = self._get_reader()
        results = reader.readtext(image_path, detail=1, paragraph=False)
 
        if not results:
            return "", 0.0
 
        text_parts = []
        confidences = []
 
        for (bbox, text, conf) in results:
            text_parts.append(text)
            confidences.append(conf)
 
        raw_text = "\n".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
 
        return raw_text, avg_confidence
 
    @property
    def engine_name(self) -> str:
        return "easyocr"

 
class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR engine (requires pytesseract + tesseract-ocr installed).
    Good for clean, printed text on white backgrounds.
    """
 
    def extract_text(self, image_path: str) -> tuple[str, float]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise OCRError("pytesseract not installed. Run: pip install pytesseract")
 
        img = Image.open(image_path)

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(img)

        confs = [c for c in data["conf"] if c != -1]
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
 
        return text, avg_conf
 
    @property
    def engine_name(self) -> str:
        return "tesseract"
 
 
class StubOCREngine(BaseOCREngine):
    """
    Returns realistic mock OCR output for testing and demos.
    Activated when OCR_ENGINE=stub in settings.
    """
 
    MOCK_RECEIPTS = [
        """
WALMART SUPERCENTER
123 Main Street, Austin TX 78701
Tel: (512) 555-0123
 
Date: 2024-03-15    Time: 14:32
Receipt #: WMT-2024-087234
 
GROCERY ITEMS
--------------
Whole Milk 1 Gal          $3.49
Bread Wheat 20oz          $2.99
Chicken Breast 2.5lb      $8.75
Orange Juice 1.75L        $4.29
Greek Yogurt 32oz         $5.49
Pasta Spaghetti           $1.89
Tomato Sauce 24oz         $2.49
Eggs Large 12ct           $4.99
 
NON-GROCERY
-----------
Shampoo Pantene           $6.99
Notebook 3pk              $4.49
USB Cable                 $9.99
 
--------------
Subtotal:               $55.85
Tax (8.25%):             $4.61
Total:                  $60.46
 
Payment: VISA *4521
Auth: 082341
 
Thank you for shopping!
Have a great day!
""",
        """
UBER RECEIPT
Trip completed: March 15, 2024
 
From: Airport Terminal 2
To: Downtown Hotel
 
Distance: 18.3 miles
Duration: 32 minutes
 
Base Fare:              $12.00
Distance (18.3mi):      $18.30
Time (32min):            $8.00
Booking Fee:             $2.99
Subtotal:               $41.29
Surge (1.2x):            $8.26
Total:                  $49.55
 
Payment: Mastercard *7823
 
Trip ID: UBR-20240315-5522
""",
        """
McDONALD'S #04521
1400 Airport Blvd
Austin, TX 78702
 
03/15/2024 12:45 PM
Order #: 847
 
Big Mac Meal L           $9.49
  Coca-Cola L
  Lg Fries
Quarter Pounder Meal     $10.29
McFlurry Oreo            $4.39
Apple Pie 2pk            $1.99
 
Subtotal:               $26.16
Tax:                     $2.16
Total:                  $28.32
 
Cash Tendered:          $30.00
Change:                  $1.68
 
Thank you! Come again!
"""
    ]
 
    _call_count = 0
 
    def extract_text(self, image_path: str) -> tuple[str, float]:
        import random
        idx = StubOCREngine._call_count % len(self.MOCK_RECEIPTS)
        StubOCREngine._call_count += 1
        return self.MOCK_RECEIPTS[idx].strip(), random.uniform(0.82, 0.96)
 
    @property
    def engine_name(self) -> str:
        return "stub"
 

 
class OCREngineFactory:
    """Creates the configured OCR engine."""
 
    _engines = {
        "easyocr": EasyOCREngine,
        "tesseract": TesseractEngine,
        "stub": StubOCREngine,
    }
 
    @classmethod
    def create(cls, engine_name: Optional[str] = None) -> BaseOCREngine:
        name = engine_name or settings.OCR_ENGINE
        engine_cls = cls._engines.get(name)
        if not engine_cls:
            raise OCRError(f"Unknown OCR engine: {name}. Choose from: {list(cls._engines.keys())}")
        return engine_cls()

 
class OCRService:
    """
    Orchestrates OCR processing with preprocessing and postprocessing.
    This is the main entry point for the OCR pipeline stage.
    """
 
    def __init__(self, engine: Optional[BaseOCREngine] = None):
        self.engine = engine or OCREngineFactory.create()
        logger.info(f"OCRService initialized with engine: {self.engine.engine_name}")
 
    def preprocess_image(self, image_path: str) -> str:
        """
        Image preprocessing to improve OCR accuracy.
        Returns path to preprocessed image.
        """
        try:
            from PIL import Image, ImageFilter, ImageEnhance
            import os
 
            img = Image.open(image_path)

            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            w, h = img.size
            if w < 800 or h < 800:
                scale = max(800 / w, 800 / h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            img = img.filter(ImageFilter.SHARPEN)
 
            base, ext = os.path.splitext(image_path)
            preprocessed_path = f"{base}_preprocessed{ext}"
            img.save(preprocessed_path)
 
            return preprocessed_path
 
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image_path
 
    def process(self, image_path: str) -> OCRResult:
        """
        Full OCR pipeline: preprocess → extract → return structured result.
        """
        if not Path(image_path).exists():
            raise OCRError(f"Image file not found: {image_path}")
 
        start = time.time()

        try:
            processed_path = self.preprocess_image(image_path)
        except Exception:
            processed_path = image_path

        try:
            raw_text, confidence = self.engine.extract_text(processed_path)
        except Exception as e:
            raise OCRError(f"OCR extraction failed: {str(e)}")
 
        duration_ms = (time.time() - start) * 1000
 
        logger.info(
            f"OCR complete | engine={self.engine.engine_name} "
            f"| confidence={confidence:.2%} | time={duration_ms:.0f}ms"
        )
 
        return OCRResult(
            raw_text=raw_text,
            confidence=round(confidence, 4),
            engine=self.engine.engine_name,
            processing_time_ms=round(duration_ms, 2)
        )
 

_ocr_service: Optional[OCRService] = None
 
 
def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service