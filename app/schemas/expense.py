from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
 
 
class ExpenseCategory(str, Enum):
    FOOD_DINING = "food_dining"
    GROCERIES = "groceries"
    TRANSPORT = "transport"
    FUEL = "fuel"
    HEALTHCARE = "healthcare"
    PHARMACY = "pharmacy"
    ENTERTAINMENT = "entertainment"
    CLOTHING = "clothing"
    ELECTRONICS = "electronics"
    UTILITIES = "utilities"
    EDUCATION = "education"
    TRAVEL = "travel"
    ACCOMMODATION = "accommodation"
    PERSONAL_CARE = "personal_care"
    STATIONERY = "stationery"
    OTHERS = "others"
 
 
class ConfidenceLevel(str, Enum):
    HIGH = "high"       
    MEDIUM = "medium"   
    LOW = "low"         
 
 
class LineItemBase(BaseModel):
    name: str = Field(..., description="Item name extracted from receipt")
    quantity: Optional[float] = Field(1.0, description="Item quantity")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    total_price: float = Field(..., description="Total price for this line item")
    category: ExpenseCategory = Field(default=ExpenseCategory.OTHERS)
    category_confidence: float = Field(0.0, ge=0.0, le=1.0)
 
 
class LineItemResponse(LineItemBase):
    id: Optional[int] = None
 
    class Config:
        from_attributes = True
 
 
class OCRResult(BaseModel):
    raw_text: str = Field(..., description="Full raw OCR output")
    confidence: float = Field(..., ge=0.0, le=1.0)
    engine: str = Field(..., description="OCR engine used")
    processing_time_ms: float
 
 
class ParsedReceipt(BaseModel):
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    items: List[LineItemBase] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None
    invoice_number: Optional[str] = None
    raw_text: str = ""
 
 
class CategoryBreakdown(BaseModel):
    category: ExpenseCategory
    total_amount: float
    item_count: int
    percentage: float
    items: List[str] = []
 
 
class ExpenseUploadResponse(BaseModel):
    expense_id: int
    status: str = "processed"
    file_name: str
    ocr_confidence: float
    confidence_level: ConfidenceLevel
 
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    payment_method: Optional[str] = None
 
    items: List[LineItemResponse] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total_amount: float = 0.0
 
    category_breakdown: List[CategoryBreakdown] = []
    primary_category: Optional[ExpenseCategory] = None
 
    processing_time_ms: float
    created_at: datetime
 
    class Config:
        from_attributes = True
 
 
class ExpenseSummary(BaseModel):
    id: int
    file_name: str
    merchant_name: Optional[str]
    total_amount: float
    primary_category: Optional[ExpenseCategory]
    ocr_confidence: float
    created_at: datetime
 
    class Config:
        from_attributes = True
 
 
class ExpenseListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    expenses: List[ExpenseSummary]
 
 
class SpendingTrendPoint(BaseModel):
    date: str
    amount: float
    count: int
 
 
class AnalyticsResponse(BaseModel):
    total_expenses: int
    total_amount: float
    avg_expense_amount: float
    category_breakdown: List[CategoryBreakdown]
    top_merchants: List[Dict[str, Any]]
    spending_trend: List[SpendingTrendPoint]
    period_days: int
 
 
class PipelineStage(BaseModel):
    stage: str
    status: str  
    duration_ms: Optional[float] = None
    error: Optional[str] = None
 
 
class ProcessingPipelineResult(BaseModel):
    stages: List[PipelineStage]
    total_duration_ms: float
    success: bool
    parsed_receipt: Optional[ParsedReceipt] = None
    ocr_result: Optional[OCRResult] = None