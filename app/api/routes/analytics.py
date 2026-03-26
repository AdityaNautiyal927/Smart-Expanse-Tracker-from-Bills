from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.services.expense_service import ExpenseProcessingService, get_expense_service
from app.schemas.expense import AnalyticsResponse
 
router = APIRouter()
 
 
@router.get(
    "/",
    response_model=AnalyticsResponse,
    summary="Spending analytics",
    description="Category-wise breakdown, top merchants, and spending trends"
)
async def get_analytics(
    days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    db: AsyncSession = Depends(get_db),
    service: ExpenseProcessingService = Depends(get_expense_service),
):
    return await service.get_analytics(db=db, days=days)
 