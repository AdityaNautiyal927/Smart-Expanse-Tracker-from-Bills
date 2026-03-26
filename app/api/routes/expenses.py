import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.services.expense_service import ExpenseProcessingService, get_expense_service
from app.utils.file_utils import save_upload
from app.schemas.expense import ExpenseUploadResponse, ExpenseListResponse
from app.core.exceptions import (
    ExpenseTrackerError, FileValidationError, OCRError, ParsingError
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/upload",
    response_model=ExpenseUploadResponse,
    summary="Upload a bill/receipt image",
    description="""
    Upload a receipt or bill image (JPG, PNG, WebP, PDF) and get back:
    - Itemized expense list with prices
    - Auto-detected categories per item
    - Category-wise spending breakdown
    - Merchant, date, payment method extraction
    - Total, subtotal, tax breakdown
    """
)
async def upload_expense(
    file: UploadFile = File(..., description="Receipt/bill image (JPG, PNG, WebP, BMP, TIFF, PDF)"),
    db: AsyncSession = Depends(get_db),
    service: ExpenseProcessingService = Depends(get_expense_service),
):

    logger.info(f"Upload received: {file.filename} ({file.content_type})")

    try:
        file_path, file_name, file_size, mime_type = await save_upload(file)

        result = await service.process_bill(
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            mime_type=mime_type,
            db=db
        )
        return result

    except FileValidationError as e:
        raise HTTPException(status_code=422, detail={"code": e.code, "message": e.message})
    except OCRError as e:
        raise HTTPException(status_code=422, detail={"code": e.code, "message": e.message})
    except ParsingError as e:
        raise HTTPException(status_code=422, detail={"code": e.code, "message": e.message})
    except ExpenseTrackerError as e:
        raise HTTPException(status_code=500, detail={"code": e.code, "message": e.message})
    except Exception as e:
        logger.exception(f"Unexpected error processing {file.filename}")
        raise HTTPException(status_code=500, detail={"code": "UNEXPECTED_ERROR", "message": str(e)})


@router.get(
    "/",
    response_model=ExpenseListResponse,
    summary="List all expenses"
)
async def list_expenses(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    service: ExpenseProcessingService = Depends(get_expense_service),
):
    return await service.list_expenses(db=db, page=page, page_size=page_size)


@router.get(
    "/{expense_id}",
    response_model=ExpenseUploadResponse,
    summary="Get expense detail"
)
async def get_expense(
    expense_id: int,
    db: AsyncSession = Depends(get_db),
    service: ExpenseProcessingService = Depends(get_expense_service),
):
    result = await service.get_expense(expense_id=expense_id, db=db)
    if not result:
        raise HTTPException(status_code=404, detail=f"Expense {expense_id} not found")
    return result


@router.delete(
    "/{expense_id}",
    summary="Delete an expense"
)
async def delete_expense(
    expense_id: int,
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select, delete
    from app.models.expense import Expense, ExpenseItem, CategoryBreakdown

    result = await db.execute(select(Expense).where(Expense.id == expense_id))
    expense = result.scalar_one_or_none()
    if not expense:
        raise HTTPException(status_code=404, detail=f"Expense {expense_id} not found")

    await db.execute(delete(ExpenseItem).where(ExpenseItem.expense_id == expense_id))
    await db.execute(delete(CategoryBreakdown).where(CategoryBreakdown.expense_id == expense_id))
    await db.delete(expense)
    await db.commit()

    return {"message": f"Expense {expense_id} deleted successfully"}