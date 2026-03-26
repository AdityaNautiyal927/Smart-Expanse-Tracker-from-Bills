import re
import logging
from typing import Optional, List, Tuple
from app.schemas.expense import ParsedReceipt, LineItemBase, ExpenseCategory
from app.core.exceptions import ParsingError
 
logger = logging.getLogger(__name__)
 

PRICE_PATTERN = re.compile(
    r"(?:[$£€₹₩¥]|Rs\.?|INR)?\s*(\d{1,4}(?:[,\.]\d{3})*(?:[.,]\d{2})?)",
    re.IGNORECASE
)

LINE_PRICE_PATTERN = re.compile(
    r"^(.+?)\s{2,}(?:[$£€₹]?\s*)?(\d{1,4}[.,]\d{2})\s*$"
)

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
    re.compile(r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b"),
    re.compile(r"\b(\w+ \d{1,2},?\s*\d{4})\b"),
    re.compile(r"\b(\d{1,2}\s+\w+\s+\d{4})\b"),
]

TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\b")

INVOICE_PATTERN = re.compile(
    r"(?:invoice|receipt|order|bill|txn|transaction|ref|#)\s*[#:]?\s*([A-Z0-9\-]+)",
    re.IGNORECASE
)
 
PAYMENT_PATTERNS = [
    (re.compile(r"\b(visa|mastercard|amex|american express)\b", re.I), lambda m: m.group(1).title()),
    (re.compile(r"\b(cash|debit|credit|upi|gpay|phonepe|paytm|neft|rtgs)\b", re.I), lambda m: m.group(1).upper()),
    (re.compile(r"\*(\d{4})\b"), lambda m: f"Card *{m.group(1)}"),
]

TOTAL_KEYWORDS = re.compile(
    r"^\s*(subtotal|sub[\s-]?total|total|grand\s+total|amount\s+due|"
    r"balance|net\s+total|amount\s+payable|tax|gst|vat|cgst|sgst|"
    r"discount|savings|change|cash\s+tendered|tips?|service\s+charge|"
    r"delivery|delivery\s+fee|shipping|handling)\b",
    re.IGNORECASE
)

HEADER_KEYWORDS = re.compile(
    r"^\s*(thank\s+you|welcome|receipt|invoice|bill\s+to|date|time|"
    r"order|cashier|server|table|tel|phone|www\.|http|address|"
    r"pos\s+#|terminal|auth|approved|signature|item\s+qty|"
    r"description\s+qty|qty\s+price|qty\s+amount)\b",
    re.IGNORECASE
)

QUANTITY_PATTERN = re.compile(
    r"^(\d+)\s*[xX×@]\s*(.+?)(?:\s{2,})([\d.,]+)\s*$"
    r"|^(.+?)\s+(\d+)\s+(?:[@x×])\s+([\d.,]+)\s*$"
)
 
 
 
class ReceiptParser:
    """
    Stateless parser: text in → ParsedReceipt out.
    No external dependencies - pure Python regex/heuristics.
    """
 
    def parse(self, raw_text: str) -> ParsedReceipt:
        if not raw_text or not raw_text.strip():
            raise ParsingError("Empty OCR text - nothing to parse")
 
        lines = self._clean_lines(raw_text)
 
        receipt = ParsedReceipt(raw_text=raw_text)
 
        receipt.merchant_name = self._extract_merchant(lines)
        receipt.date = self._extract_date(raw_text)
        receipt.time = self._extract_time(raw_text)
        receipt.invoice_number = self._extract_invoice(raw_text)
        receipt.payment_method = self._extract_payment_method(raw_text)
 
        receipt.subtotal, receipt.tax, receipt.discount, receipt.total = (
            self._extract_financials(lines)
        )
 
        receipt.items = self._extract_items(lines)
 
        if receipt.total is None and receipt.items:
            receipt.total = round(sum(i.total_price for i in receipt.items), 2)
 
        logger.info(
            f"Parsed receipt | merchant={receipt.merchant_name} | "
            f"items={len(receipt.items)} | total={receipt.total}"
        )
        return receipt

 
    def _clean_lines(self, text: str) -> List[str]:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if re.match(r"^[-=_*\.]{3,}$", line):
                continue
            if not line:
                continue
            cleaned.append(line)
        return cleaned

 
    def _extract_merchant(self, lines: List[str]) -> Optional[str]:
        """
        Merchant is typically the first non-noise line (2-5 words, no price).
        """
        for line in lines[:6]:
            if PRICE_PATTERN.search(line):
                continue
            if HEADER_KEYWORDS.match(line):
                continue
            if TOTAL_KEYWORDS.match(line):
                continue
            if re.search(r"[A-Za-z]", line) and 3 <= len(line) <= 80:
                return line.strip()
        return None

 
    def _extract_date(self, text: str) -> Optional[str]:
        for pattern in DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(1)
        return None
 
    def _extract_time(self, text: str) -> Optional[str]:
        m = TIME_PATTERN.search(text)
        return m.group(1) if m else None
 
 
    def _extract_invoice(self, text: str) -> Optional[str]:
        m = INVOICE_PATTERN.search(text)
        return m.group(1) if m else None
 
 
    def _extract_payment_method(self, text: str) -> Optional[str]:
        for pattern, formatter in PAYMENT_PATTERNS:
            m = pattern.search(text)
            if m:
                return formatter(m)
        return None
 
 
    def _extract_financials(self, lines: List[str]) -> Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float]
    ]:
        subtotal = tax = discount = total = None
 
        for line in lines:
            lower = line.lower()
            prices = self._extract_prices_from_line(line)
            if not prices:
                continue
 
            price = prices[-1] 
 
            if re.search(r"\b(grand\s+total|total\s+amount|amount\s+due|total)\b", lower):
                total = price
            elif re.search(r"\b(subtotal|sub\s+total)\b", lower):
                subtotal = price
            elif re.search(r"\b(tax|gst|vat|cgst|sgst|cess)\b", lower):
                tax = (tax or 0) + price
            elif re.search(r"\b(discount|savings|offer|promo)\b", lower):
                discount = (discount or 0) + price
 
        return subtotal, tax, discount, total
 
 
    def _extract_items(self, lines: List[str]) -> List[LineItemBase]:
        items = []
 
        for line in lines:
            if TOTAL_KEYWORDS.match(line):
                continue
            if HEADER_KEYWORDS.match(line):
                continue
 
            item = self._parse_item_line(line)
            if item:
                items.append(item)
 
        return items
 
    def _parse_item_line(self, line: str) -> Optional[LineItemBase]:
        """Try multiple patterns to parse an item line."""
 
        qty_match = re.match(
            r"^(.+?)\s+(\d+(?:\.\d+)?)\s*[xX×@]\s*([\d.,]+)\s{1,}([\d.,]+)\s*$",
            line
        )
        if qty_match:
            name = qty_match.group(1).strip()
            qty = float(qty_match.group(2))
            unit_price = self._parse_price(qty_match.group(3))
            total = self._parse_price(qty_match.group(4))
            if name and total is not None and total > 0:
                return LineItemBase(
                    name=self._clean_item_name(name),
                    quantity=qty,
                    unit_price=unit_price,
                    total_price=total,
                    category=ExpenseCategory.OTHERS
                )

        basic_match = LINE_PRICE_PATTERN.match(line)
        if basic_match:
            name = basic_match.group(1).strip()
            price_str = basic_match.group(2)
            price = self._parse_price(price_str)
 
            if name and price is not None and price > 0:
                if TOTAL_KEYWORDS.match(name) or HEADER_KEYWORDS.match(name):
                    return None
                if len(name) < 2 or re.match(r"^[\d\s.,]+$", name):
                    return None
 
                return LineItemBase(
                    name=self._clean_item_name(name),
                    quantity=1.0,
                    unit_price=price,
                    total_price=price,
                    category=ExpenseCategory.OTHERS
                )
 
        # Pattern 3: Price at end after tab/multiple spaces
        tab_match = re.match(r"^(.+?)[\t]([\d.,]+)\s*$", line)
        if tab_match:
            name = tab_match.group(1).strip()
            price = self._parse_price(tab_match.group(2))
            if name and price and price > 0 and len(name) > 2:
                if not TOTAL_KEYWORDS.match(name) and not HEADER_KEYWORDS.match(name):
                    return LineItemBase(
                        name=self._clean_item_name(name),
                        quantity=1.0,
                        unit_price=price,
                        total_price=price,
                        category=ExpenseCategory.OTHERS)
 
        return None
 
    def _clean_item_name(self, name: str) -> str:
        """Clean up item names: remove codes, normalize whitespace."""
        name = re.sub(r"^[A-Z]?\d{2,6}\s+", "", name)
        name = re.sub(r"[*#@]+$", "", name)
        name = " ".join(name.split())
        return name.strip()
 
    def _parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float."""
        if not price_str:
            return None
        cleaned = re.sub(r"[$£€₹₩¥]|Rs\.?|INR", "", price_str).strip()
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
 
        try:
            val = float(cleaned)
            if 0 < val < 1_000_000:
                return round(val, 2)
        except ValueError:
            pass
        return None
 
    def _extract_prices_from_line(self, line: str) -> List[float]:
        """Extract all prices from a line."""
        results = []
        for m in PRICE_PATTERN.finditer(line):
            price = self._parse_price(m.group(1))
            if price:
                results.append(price)
        return results
 
 
class ParserService:
    """Wraps ReceiptParser with timing and logging."""
 
    def __init__(self):
        self.parser = ReceiptParser()
 
    def process(self, raw_text: str) -> ParsedReceipt:
        import time
        start = time.time()
        result = self.parser.parse(raw_text)
        duration = (time.time() - start) * 1000
        logger.info(f"Parser completed in {duration:.1f}ms")
        return result
 
 
def get_parser_service() -> ParserService:
    return ParserService()