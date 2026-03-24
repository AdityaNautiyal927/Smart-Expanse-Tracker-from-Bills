import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
 
from app.schemas.expense import ExpenseCategory, LineItemBase, CategoryBreakdown
from app.core.config import settings
 
logger = logging.getLogger(__name__)
 

CATEGORY_RULES: Dict[ExpenseCategory, List[str]] = {
    ExpenseCategory.FOOD_DINING: [
        "burger", "pizza", "sandwich", "coffee", "tea", "latte", "cappuccino",
        "espresso", "meal", "lunch", "dinner", "breakfast", "restaurant",
        "cafe", "diner", "wings", "sushi", "noodle", "pasta", "steak",
        "salad", "soup", "combo", "mcmuffin", "whopper", "wrap", "taco",
        "burrito", "fries", "nuggets", "mcflurry", "smoothie", "juice bar",
        "donut", "bagel", "muffin", "croissant", "cookie", "cake slice",
        "big mac", "quarter pounder", "zinger", "grilled",
    ],
    ExpenseCategory.GROCERIES: [
        "milk", "bread", "butter", "cheese", "egg", "flour", "sugar", "salt",
        "rice", "dal", "lentil", "vegetable", "fruit", "apple", "banana",
        "orange", "tomato", "potato", "onion", "garlic", "ginger", "oil",
        "ghee", "yogurt", "curd", "cream", "paneer", "chicken", "fish",
        "mutton", "beef", "pork", "pasta", "noodle", "sauce", "ketchup",
        "mustard", "mayonnaise", "jam", "honey", "cereal", "oats",
        "cornflakes", "biscuit", "cracker", "chips", "snack", "popcorn",
        "soft drink", "cola", "water bottle", "juice", "soda", "mineral water",
        "grocery", "supermarket", "walmart", "target", "whole foods",
    ],
    ExpenseCategory.TRANSPORT: [
        "uber", "lyft", "ola", "taxi", "cab", "ride", "auto", "rickshaw",
        "bus", "metro", "subway", "train", "rail", "tram", "ferry",
        "ticket", "fare", "transport", "commute", "shuttle", "rapido",
        "booking fee", "surge", "base fare", "trip",
    ],
    ExpenseCategory.FUEL: [
        "petrol", "diesel", "fuel", "gas station", "gasoline", "cng",
        "lpg", "pump", "liter", "litre", "shell", "bp", "chevron",
        "hpcl", "iocl", "bpcl", "indian oil", "bharat petroleum",
    ],
    ExpenseCategory.HEALTHCARE: [
        "hospital", "clinic", "doctor", "physician", "consultation",
        "medical", "health", "dental", "dentist", "eye care", "optician",
        "lab test", "blood test", "xray", "x-ray", "scan", "mri", "ct scan",
        "physiotherapy", "therapy", "surgery", "operation",
    ],
    ExpenseCategory.PHARMACY: [
        "medicine", "tablet", "capsule", "syrup", "drops", "cream", "ointment",
        "pharmacy", "chemist", "drug store", "medplus", "apollo pharmacy",
        "1mg", "netmeds", "paracetamol", "ibuprofen", "antibiotic",
        "vitamin", "supplement", "bandage", "first aid",
    ],
    ExpenseCategory.ENTERTAINMENT: [
        "movie", "cinema", "theatre", "concert", "show", "event",
        "netflix", "spotify", "amazon prime", "hotstar", "disney",
        "game", "gaming", "ps5", "xbox", "steam", "app store",
        "streaming", "subscription", "ticket", "amusement", "bowling",
        "bowling", "laser tag", "escape room", "museum", "zoo",
    ],
    ExpenseCategory.CLOTHING: [
        "shirt", "tshirt", "t-shirt", "trouser", "pant", "jean", "denim",
        "dress", "skirt", "blouse", "jacket", "coat", "sweater", "hoodie",
        "shoes", "sneaker", "boot", "sandal", "slipper", "sock",
        "underwear", "bra", "lingerie", "tie", "belt", "bag", "purse",
        "handbag", "wallet", "cap", "hat", "scarf", "gloves",
        "zara", "h&m", "forever21", "myntra", "ajio",
    ],
    ExpenseCategory.ELECTRONICS: [
        "phone", "mobile", "laptop", "computer", "tablet", "ipad",
        "charger", "usb cable", "usb hub", "usb drive", "type c cable",
        "hdmi cable", "lightning cable", "adapter", "earphone", "headphone",
        "speaker", "camera", "battery", "memory card", "sd card",
        "keyboard", "mouse", "monitor", "tv", "television", "remote",
        "router", "modem", "printer", "ink", "toner", "hard disk", "ssd",
    ],
    ExpenseCategory.UTILITIES: [
        "electricity", "electric bill", "water bill", "gas bill",
        "internet", "broadband", "wifi", "mobile recharge", "recharge",
        "postpaid", "airtel", "jio", "vodafone", "bsnl",
        "maintenance", "society", "rent", "emi",
    ],
    ExpenseCategory.EDUCATION: [
        "book", "textbook", "notebook", "stationery", "pen", "pencil",
        "tuition", "course", "class", "school", "college", "university",
        "fee", "exam", "coaching", "training", "workshop", "seminar",
        "udemy", "coursera", "skillshare",
    ],
    ExpenseCategory.TRAVEL: [
        "flight", "airline", "airways", "airport", "hotel", "resort",
        "hostel", "airbnb", "booking.com", "makemytrip", "yatra",
        "cleartrip", "visa", "passport", "travel insurance",
        "luggage", "suitcase", "boarding pass",
    ],
    ExpenseCategory.ACCOMMODATION: [
        "hotel", "motel", "inn", "lodge", "resort", "hostel", "room",
        "suite", "accommodation", "stay", "check-in", "check-out",
        "night charge", "room service",
    ],
    ExpenseCategory.PERSONAL_CARE: [
        "shampoo", "conditioner", "soap", "facewash", "moisturizer",
        "lotion", "sunscreen", "deodorant", "perfume", "cologne",
        "razor", "shaving", "toothbrush", "toothpaste", "mouthwash",
        "cotton", "tissue", "wipes", "makeup", "lipstick", "foundation",
        "mascara", "nail polish", "salon", "haircut", "spa",
    ],
    ExpenseCategory.STATIONERY: [
        "pen", "pencil", "eraser", "ruler", "stapler", "tape", "glue",
        "paper", "notepad", "folder", "file", "binder", "highlighter",
        "marker", "whiteboard", "notebook",
    ],
}
 
# Precompile keyword → category for fast O(1) lookup
_KEYWORD_MAP: Dict[str, ExpenseCategory] = {}
for _cat, _keywords in CATEGORY_RULES.items():
    for _kw in _keywords:
        _KEYWORD_MAP[_kw.lower()] = _cat
 

 
class BaseCategorizer(ABC):
    @abstractmethod
    def categorize(self, item_name: str) -> Tuple[ExpenseCategory, float]:
        """Returns (category, confidence 0-1)."""
        ...

 
class RuleBasedCategorizer(BaseCategorizer):
 
    def categorize(self, item_name: str) -> Tuple[ExpenseCategory, float]:
        name_lower = item_name.lower().strip()

        best_cat = None
        best_len = 0
        best_conf = 0.0
 
        for kw, cat in sorted(_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if kw in name_lower:
                kw_len = len(kw)
                if kw_len > best_len:
                    best_len = kw_len
                    best_cat = cat
                    best_conf = 0.95 if kw_len > 4 else 0.80
 
        if best_cat is not None:
            return best_cat, best_conf
 
        words = re.findall(r"\b\w+\b", name_lower)
        for word in words:
            if len(word) < 4:
                continue
            for kw, cat in _KEYWORD_MAP.items():
                kw_parts = kw.split()
                if len(kw_parts) == 1 and len(kw) >= 4:
                    if word.startswith(kw[:4]) or kw.startswith(word[:4]):
                        return cat, 0.65
 
        return ExpenseCategory.OTHERS, 0.50
 

 
class MLCategorizer(BaseCategorizer):
 
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._fallback = RuleBasedCategorizer()
        self._load_model()
 
    def _load_model(self):
        import os
        model_path = settings.ML_MODEL_PATH
        if not os.path.exists(model_path):
            logger.info("ML model not found, will use rule-based fallback")
            return
 
        try:
            import joblib
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.vectorizer = bundle["vectorizer"]
            self.label_encoder = bundle["label_encoder"]
            logger.info(f"ML categorizer loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
 
    def categorize(self, item_name: str) -> Tuple[ExpenseCategory, float]:
        if self.model is None:
            return self._fallback.categorize(item_name)
 
        try:
            features = self.vectorizer.transform([item_name.lower()])
            proba = self.model.predict_proba(features)[0]
            pred_idx = proba.argmax()
            confidence = float(proba[pred_idx])
            label = self.label_encoder.inverse_transform([pred_idx])[0]
            category = ExpenseCategory(label)
            return category, confidence
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}, falling back to rules")
            return self._fallback.categorize(item_name)
 

 
class HybridCategorizer(BaseCategorizer):
    """
    Rules first (fast + interpretable).
    Defers to ML only when rule confidence is low (< 0.7).
    """
 
    def __init__(self):
        self.rules = RuleBasedCategorizer()
        self.ml = MLCategorizer()
 
    def categorize(self, item_name: str) -> Tuple[ExpenseCategory, float]:
        cat, conf = self.rules.categorize(item_name)
        if conf >= 0.70:
            return cat, conf
        ml_cat, ml_conf = self.ml.categorize(item_name)
        if ml_conf > conf:
            return ml_cat, ml_conf
        return cat, conf
 
 
class CategorizationService:
    """
    Applies categorization to all items and produces breakdown.
    """
 
    def __init__(self):
        engine = settings.CATEGORIZATION_ENGINE
        engines = {
            "rule_based": RuleBasedCategorizer,
            "ml_model": MLCategorizer,
            "hybrid": HybridCategorizer,
        }
        cls = engines.get(engine, RuleBasedCategorizer)
        self.categorizer = cls()
        logger.info(f"CategorizationService using: {cls.__name__}")
 
    def categorize_items(self, items: List[LineItemBase]) -> List[LineItemBase]:
        """Assign category and confidence to each item."""
        for item in items:
            cat, conf = self.categorizer.categorize(item.name)
            item.category = cat
            item.category_confidence = round(conf, 4)
        return items
 
    def compute_breakdown(
        self,
        items: List[LineItemBase],
        total: Optional[float] = None
    ) -> List[CategoryBreakdown]:
        """
        Aggregate items into category-wise breakdown.
        Returns sorted list (highest spend first).
        """
        agg: Dict[str, Dict] = {}
 
        for item in items:
            cat = item.category.value
            if cat not in agg:
                agg[cat] = {"total": 0.0, "count": 0, "names": []}
            agg[cat]["total"] += item.total_price
            agg[cat]["count"] += 1
            agg[cat]["names"].append(item.name)
 
        grand_total = total or sum(item.total_price for item in items)
 
        breakdowns = []
        for cat, data in agg.items():
            pct = (data["total"] / grand_total * 100) if grand_total > 0 else 0.0
            breakdowns.append(CategoryBreakdown(
                category=ExpenseCategory(cat),
                total_amount=round(data["total"], 2),
                item_count=data["count"],
                percentage=round(pct, 2),
                items=data["names"][:5],  
            ))
 
        return sorted(breakdowns, key=lambda x: x.total_amount, reverse=True)
 
    def get_primary_category(
        self, breakdown: List[CategoryBreakdown]
    ) -> Optional[ExpenseCategory]:
        if not breakdown:
            return None
        return breakdown[0].category
 
 
def get_categorization_service() -> CategorizationService:
    return CategorizationService()