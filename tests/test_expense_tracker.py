import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport

class TestReceiptParser:

    def setup_method(self):
        from app.services.parser.parser_service import ReceiptParser
        self.parser = ReceiptParser()

    def test_parse_walmart_receipt(self):
        text = """
WALMART SUPERCENTER
123 Main Street, Austin TX

Date: 03/15/2024    Time: 14:32
Receipt #: WMT-2024-087234

Whole Milk 1 Gal          $3.49
Bread Wheat 20oz          $2.99
Chicken Breast 2.5lb      $8.75
Greek Yogurt 32oz         $5.49

Subtotal:               $20.72
Tax (8.25%):             $1.71
Total:                  $22.43

Payment: VISA *4521
"""
        receipt = self.parser.parse(text)
        assert receipt.merchant_name == "WALMART SUPERCENTER"
        assert receipt.date == "03/15/2024"
        assert receipt.time == "14:32"
        assert receipt.invoice_number == "WMT-2024-087234"
        assert receipt.payment_method is not None
        assert len(receipt.items) >= 3
        assert receipt.total == 22.43
        assert receipt.tax == 1.71
        assert receipt.subtotal == 20.72

    def test_parse_uber_receipt(self):
        text = """
UBER RECEIPT
Trip completed: March 15, 2024

From: Airport Terminal 2
To: Downtown Hotel

Base Fare:              $12.00
Distance:               $18.30
Time:                    $8.00
Booking Fee:             $2.99
Total:                  $41.29

Payment: Mastercard *7823
Trip ID: UBR-20240315-5522
"""
        receipt = self.parser.parse(text)
        assert receipt.merchant_name is not None
        assert receipt.total == 41.29
        assert receipt.payment_method is not None

    def test_parse_price_formats(self):
        """Test various price format handling."""
        prices = [
            ("$3.49", 3.49),
            ("12.99", 12.99),
            ("1,234.56", 1234.56),
            ("₹150.00", 150.00),
            ("£9.99", 9.99),
        ]
        for price_str, expected in prices:
            result = self.parser._parse_price(price_str)
            assert result == expected, f"Failed for {price_str}: got {result}"

    def test_empty_text_raises(self):
        from app.core.exceptions import ParsingError
        with pytest.raises(ParsingError):
            self.parser.parse("")

    def test_extract_dates(self):
        texts = [
            ("Date: 03/15/2024", "03/15/2024"),
            ("Date: 2024-03-15", "2024-03-15"),
            ("March 15, 2024", "March 15, 2024"),
        ]
        for text, expected in texts:
            result = self.parser._extract_date(text)
            assert result == expected, f"Expected {expected}, got {result} for '{text}'"

    def test_item_with_quantity(self):
        lines = ["Coke 2 x 1.50   3.00"]
        items = self.parser._extract_items(lines)
        assert len(items) == 1
        assert items[0].quantity == 2.0
        assert items[0].total_price == 3.00

    def test_skips_total_lines(self):
        lines = [
            "Apple Juice         2.99",
            "Total:             12.99",
            "Subtotal:          10.00",
            "Tax:                2.99",
        ]
        items = self.parser._extract_items(lines)
        assert len(items) == 1
        assert "Apple" in items[0].name


class TestCategorizationService:
    """Unit tests for rule-based categorizer."""

    def setup_method(self):
        from app.services.categorizer.categorization_service import RuleBasedCategorizer
        from app.schemas.expense import ExpenseCategory
        self.categorizer = RuleBasedCategorizer()
        self.Category = ExpenseCategory

    def test_food_items(self):
        items = ["Big Mac", "Burger", "Coffee Latte", "Pizza Margherita"]
        for item in items:
            cat, conf = self.categorizer.categorize(item)
            assert cat == self.Category.FOOD_DINING, f"'{item}' should be FOOD_DINING, got {cat}"
            assert conf > 0.6

    def test_grocery_items(self):
        items = ["Whole Milk 1 Gal", "Bread Wheat", "Eggs Large 12ct", "Greek Yogurt"]
        for item in items:
            cat, conf = self.categorizer.categorize(item)
            assert cat == self.Category.GROCERIES, f"'{item}' should be GROCERIES, got {cat}"

    def test_transport_items(self):
        items = ["Uber Trip", "Base Fare", "Booking Fee", "Ola Ride"]
        for item in items:
            cat, conf = self.categorizer.categorize(item)
            assert cat == self.Category.TRANSPORT, f"'{item}' should be TRANSPORT"

    def test_healthcare_items(self):
        items = ["Doctor Consultation", "Hospital Visit", "Lab Test"]
        for item in items:
            cat, conf = self.categorizer.categorize(item)
            assert cat == self.Category.HEALTHCARE

    def test_unknown_returns_others(self):
        cat, conf = self.categorizer.categorize("XYZ12345 Unknown Item")
        assert cat == self.Category.OTHERS

    def test_confidence_range(self):
        items = ["milk", "bread", "uber", "coffee", "random_item_xyz"]
        for item in items:
            _, conf = self.categorizer.categorize(item)
            assert 0.0 <= conf <= 1.0, f"Confidence out of range for '{item}': {conf}"

    def test_category_breakdown(self):
        from app.services.categorizer.categorization_service import CategorizationService
        from app.schemas.expense import LineItemBase, ExpenseCategory

        service = CategorizationService()
        items = [
            LineItemBase(name="Milk", total_price=3.49, category=ExpenseCategory.OTHERS),
            LineItemBase(name="Bread", total_price=2.99, category=ExpenseCategory.OTHERS),
            LineItemBase(name="Uber Trip", total_price=15.00, category=ExpenseCategory.OTHERS),
        ]
        categorized = service.categorize_items(items)
        breakdown = service.compute_breakdown(categorized, total=21.48)

        assert len(breakdown) > 0
        total_pct = sum(b.percentage for b in breakdown)
        assert abs(total_pct - 100.0) < 1.0  # Should sum to ~100%


class TestOCRService:
    """Test OCR service with stub engine."""

    def setup_method(self):
        from app.services.ocr.ocr_service import OCRService, StubOCREngine
        self.service = OCRService(engine=StubOCREngine())

    def test_stub_returns_text(self, tmp_path):
        img_path = str(tmp_path / "test.jpg")
        try:
            from PIL import Image
            img = Image.new("RGB", (100, 100), color="white")
            img.save(img_path)
        except ImportError:
            with open(img_path, "wb") as f:
                f.write(b"\xff\xd8\xff")  

        result = self.service.process(img_path)
        assert result.raw_text != ""
        assert 0.0 <= result.confidence <= 1.0
        assert result.engine == "stub"
        assert result.processing_time_ms > 0

    def test_missing_file_raises(self):
        from app.core.exceptions import OCRError
        with pytest.raises(OCRError):
            self.service.process("/nonexistent/path/image.jpg")


class TestProcessingPipeline:
    """Integration test: stub OCR → parse → categorize."""

    def test_full_pipeline(self):
        from app.services.ocr.ocr_service import StubOCREngine, OCRService
        from app.services.parser.parser_service import ParserService
        from app.services.categorizer.categorization_service import CategorizationService
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            img_path = f.name

        try:
            ocr_service = OCRService(engine=StubOCREngine())
            ocr_result = ocr_service.process(img_path)
            assert ocr_result.raw_text

            parser = ParserService()
            parsed = parser.process(ocr_result.raw_text)
            assert parsed.items is not None

            categorizer = CategorizationService()
            items = categorizer.categorize_items(parsed.items)
            breakdown = categorizer.compute_breakdown(items, parsed.total)

            assert all(item.category is not None for item in items)
            assert all(0 <= item.category_confidence <= 1 for item in items)
            if breakdown:
                assert breakdown[0].total_amount >= 0
        finally:
            os.unlink(img_path)


@pytest.mark.asyncio
class TestExpenseAPI:
    """API-level tests using FastAPI TestClient."""

    @pytest.fixture
    async def client(self):
        import os
        os.environ["OCR_ENGINE"] = "stub"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

        from app.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as ac:
            yield ac

    async def test_health_check(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_root_endpoint(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert "Smart Expense Tracker" in response.json()["message"]

    async def test_upload_receipt(self, client, tmp_path):
        """Test full upload → OCR → parse → categorize flow."""
        img_path = tmp_path / "receipt.jpg"
        try:
            from PIL import Image
            img = Image.new("RGB", (200, 400), color="white")
            img.save(str(img_path))
        except ImportError:
            img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        with open(img_path, "rb") as f:
            response = await client.post(
                "/api/v1/expenses/upload",
                files={"file": ("receipt.jpg", f, "image/jpeg")}
            )

        assert response.status_code == 200
        data = response.json()

        assert "expense_id" in data
        assert "items" in data
        assert "total_amount" in data
        assert "category_breakdown" in data
        assert "ocr_confidence" in data
        assert "processing_time_ms" in data
        assert data["status"] == "processed"
        assert data["total_amount"] >= 0
        assert isinstance(data["items"], list)
        assert isinstance(data["category_breakdown"], list)

    async def test_list_expenses(self, client):
        response = await client.get("/api/v1/expenses/?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "expenses" in data
        assert "total" in data
        assert "page" in data

    async def test_get_nonexistent_expense(self, client):
        response = await client.get("/api/v1/expenses/99999")
        assert response.status_code == 404

    async def test_analytics_endpoint(self, client):
        response = await client.get("/api/v1/analytics/?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "total_expenses" in data
        assert "total_amount" in data
        assert "category_breakdown" in data

    async def test_upload_invalid_file(self, client):
        response = await client.post(
            "/api/v1/expenses/upload",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")}
        )
        assert response.status_code == 422

    async def test_upload_empty_file(self, client):
        response = await client.post(
            "/api/v1/expenses/upload",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 422


class TestParserEdgeCases:
    """Edge cases: noisy OCR, multi-currency, malformed input."""

    def setup_method(self):
        from app.services.parser.parser_service import ReceiptParser
        self.parser = ReceiptParser()

    def test_noisy_ocr_text(self):
        """OCR often produces garbled text; parser should still extract something."""
        noisy = """
W@LMART SUPERc3NTER
0ate: 03/15/2024

Wh0le M1lk  1 Ga|         3.49
Br3ad Wh3at              2,99
Ch1cken 8reast            8.75

T0ta1:                   15.23
"""
        receipt = self.parser.parse(noisy)
        assert receipt is not None

    def test_no_items_receipt(self):
        """Receipt with only totals and no line items."""
        text = """
QUICK MART
Date: 01/01/2024
Total:  25.00
Cash:   30.00
Change:  5.00
"""
        receipt = self.parser.parse(text)
        assert receipt.total == 25.00

    def test_indian_rupee_prices(self):
        """Test INR currency handling."""
        text = """
BIG BAZAAR
Date: 15/03/2024

Tata Salt 1kg          ₹24.00
Amul Butter 500g      ₹275.00
Britannia Bread       ₹45.00

Total:               ₹344.00
"""
        receipt = self.parser.parse(text)
        assert receipt.total == 344.00
        assert len(receipt.items) >= 2

    def test_high_value_items(self):
        """Large price values should be parsed correctly."""
        text = """
ELECTRONICS STORE
MacBook Pro 16         $2,499.00
iPhone 15 Pro          $1,199.00
AirPods Pro              $249.00

Total:               $3,947.00
"""
        receipt = self.parser.parse(text)
        assert receipt.total == 3947.00

    def test_clean_item_name(self):
        """Item names should be cleaned of artifacts."""
        names = [
            ("001 Whole Milk", "Whole Milk"),
            ("A12 Bread*", "Bread"),
            ("  Eggs   Large  ", "Eggs Large"),
        ]
        for raw, expected in names:
            result = self.parser._clean_item_name(raw)
            assert result == expected, f"Expected '{expected}', got '{result}'"



class TestMLCategorizerFallback:
    """ML categorizer should fall back gracefully when model absent."""

    def test_fallback_to_rules(self):
        from app.services.categorizer.categorization_service import MLCategorizer
        from app.schemas.expense import ExpenseCategory

        ml = MLCategorizer()
        cat, conf = ml.categorize("Whole Milk")
        assert cat == ExpenseCategory.GROCERIES
        assert conf > 0


class TestConfidenceLevels:
    def test_confidence_mapping(self):
        from app.services.expense_service import ExpenseProcessingService
        from app.schemas.expense import ConfidenceLevel

        service = ExpenseProcessingService()
        assert service._get_confidence_level(0.90) == ConfidenceLevel.HIGH
        assert service._get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert service._get_confidence_level(0.70) == ConfidenceLevel.MEDIUM
        assert service._get_confidence_level(0.60) == ConfidenceLevel.MEDIUM
        assert service._get_confidence_level(0.50) == ConfidenceLevel.LOW
        assert service._get_confidence_level(0.10) == ConfidenceLevel.LOW


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])