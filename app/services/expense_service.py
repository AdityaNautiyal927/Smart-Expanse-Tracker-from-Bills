import time
import logging
import os
import shutil
from typing import Optional
from pathlib import Path
from datetime import datetime
 
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sqlalchemy.orm import selectinload
 
from app.core.config import settings
from app.core.exceptions import FileValidationError, ExpenseTrackerError
from app.schemas.expense import (
    ExpenseUploadResponse, ParsedReceipt, LineItemResponse,
    CategoryBreakdown, ConfidenceLevel, ProcessingPipelineResult,
    PipelineStage, ExpenseListResponse, ExpenseSummary,
    AnalyticsResponse, SpendingTrendPoint, ExpenseCategory
)
from app.models.expense import Expense, ExpenseItem, CategoryBreakdown as CategoryBreakdownModel
from app.services.ocr.ocr_service import get_ocr_service
from app.services.parser.parser_service import get_parser_service
from app.services.categorizer.categorization_service import get_categorization_service
 
logger = logging.getLogger(__name__)
 
 
class ExpenseProcessingService:
    def __init__(self):
        self.ocr = get_ocr_service()
        self.parser = get_parser_service()
        self.categorizer = get_categorization_service()
 
    async def process_bill(
        self,
        file_path: str,
        file_name: str,
        file_size: int,
        mime_type: str,
        db: AsyncSession
    ) -> ExpenseUploadResponse:
        """
        Full processing pipeline:
        1. OCR: Image → raw text
        2. Parse: raw text → structured receipt
        3. Categorize: items → categories + breakdown
        4. Store: persist to database
        5. Return: structured JSON response
        """
        pipeline_start = time.time()
        stages: list[PipelineStage] = []
 
        stage_start = time.time()
        logger.info(f"[Pipeline] Stage 1: OCR | file={file_name}")
        try:
            ocr_result = self.ocr.process(file_path)
            stages.append(PipelineStage(
                stage="ocr",
                status="done",
                duration_ms=round((time.time() - stage_start) * 1000, 2)
            ))
        except Exception as e:
            stages.append(PipelineStage(stage="ocr", status="failed", error=str(e)))
            raise
 
        stage_start = time.time()
        logger.info(f"[Pipeline] Stage 2: Parse | chars={len(ocr_result.raw_text)}")
        try:
            parsed = self.parser.process(ocr_result.raw_text)
            stages.append(PipelineStage(
                stage="parse",
                status="done",
                duration_ms=round((time.time() - stage_start) * 1000, 2)
            ))
        except Exception as e:
            stages.append(PipelineStage(stage="parse", status="failed", error=str(e)))
            raise
 
        stage_start = time.time()
        logger.info(f"[Pipeline] Stage 3: Categorize | items={len(parsed.items)}")
        try:
            categorized_items = self.categorizer.categorize_items(parsed.items)
            breakdown = self.categorizer.compute_breakdown(categorized_items, parsed.total)
            primary_cat = self.categorizer.get_primary_category(breakdown)
            stages.append(PipelineStage(
                stage="categorize",
                status="done",
                duration_ms=round((time.time() - stage_start) * 1000, 2)
            ))
        except Exception as e:
            stages.append(PipelineStage(stage="categorize", status="failed", error=str(e)))
            raise
 
        stage_start = time.time()
        logger.info("[Pipeline] Stage 4: Persist")
        try:
            total_amount = parsed.total or sum(i.total_price for i in categorized_items)
            expense = await self._save_expense(
                db=db,
                file_name=file_name,
                file_path=file_path,
                file_size=file_size,
                mime_type=mime_type,
                ocr_result=ocr_result,
                parsed=parsed,
                items=categorized_items,
                breakdown=breakdown,
                primary_category=primary_cat,
                total_amount=total_amount,
                processing_time_ms=(time.time() - pipeline_start) * 1000
            )
            stages.append(PipelineStage(
                stage="persist",
                status="done",
                duration_ms=round((time.time() - stage_start) * 1000, 2)
            ))
        except Exception as e:
            stages.append(PipelineStage(stage="persist", status="failed", error=str(e)))
            raise
 
        total_duration_ms = round((time.time() - pipeline_start) * 1000, 2)
        logger.info(f"[Pipeline] Complete in {total_duration_ms:.0f}ms | expense_id={expense.id}")
 
        confidence_level = self._get_confidence_level(ocr_result.confidence)
 
        return ExpenseUploadResponse(
            expense_id=expense.id,
            status="processed",
            file_name=file_name,
            ocr_confidence=ocr_result.confidence,
            confidence_level=confidence_level,
            merchant_name=parsed.merchant_name,
            date=parsed.date,
            payment_method=parsed.payment_method,
            items=[
                LineItemResponse(
                    id=None,
                    name=item.name,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    total_price=item.total_price,
                    category=item.category,
                    category_confidence=item.category_confidence,
                )
                for item in categorized_items
            ],
            subtotal=parsed.subtotal,
            tax=parsed.tax,
            discount=parsed.discount,
            total_amount=total_amount,
            category_breakdown=breakdown,
            primary_category=primary_cat,
            processing_time_ms=total_duration_ms,
            created_at=expense.created_at or datetime.utcnow()
        )
 
    async def _save_expense(
        self, db, file_name, file_path, file_size, mime_type,
        ocr_result, parsed, items, breakdown, primary_category,
        total_amount, processing_time_ms
    ) -> Expense:
        """Persist expense, items, and breakdown to database."""
        expense = Expense(
            file_name=file_name,
            file_path=file_path,
            file_size_bytes=file_size,
            mime_type=mime_type,
            raw_ocr_text=ocr_result.raw_text,
            ocr_confidence=ocr_result.confidence,
            ocr_engine=ocr_result.engine,
            merchant_name=parsed.merchant_name,
            merchant_address=parsed.merchant_address,
            invoice_number=parsed.invoice_number,
            receipt_date=parsed.date,
            receipt_time=parsed.time,
            payment_method=parsed.payment_method,
            subtotal=parsed.subtotal,
            tax_amount=parsed.tax,
            discount_amount=parsed.discount,
            total_amount=total_amount,
            primary_category=primary_category.value if primary_category else None,
            processing_time_ms=processing_time_ms,
            status="processed",
        )
        db.add(expense)
        await db.flush()  

        for item in items:
            db_item = ExpenseItem(
                expense_id=expense.id,
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                total_price=item.total_price,
                category=item.category.value,
                category_confidence=item.category_confidence,
            )
            db.add(db_item)

        for cat in breakdown:
            db_cat = CategoryBreakdownModel(
                expense_id=expense.id,
                category=cat.category.value,
                total_amount=cat.total_amount,
                item_count=cat.item_count,
                percentage=cat.percentage,
            )
            db.add(db_cat)
 
        await db.commit()
        await db.refresh(expense)
        return expense
 
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
 
    async def get_expense(self, expense_id: int, db: AsyncSession) -> Optional[ExpenseUploadResponse]:
        """Retrieve a single processed expense."""
        result = await db.execute(
            select(Expense)
            .options(
                selectinload(Expense.items),
                selectinload(Expense.category_breakdowns)
            )
            .where(Expense.id == expense_id)
        )
        expense = result.scalar_one_or_none()
        if not expense:
            return None
        return self._expense_to_response(expense)
 
    async def list_expenses(
        self, db: AsyncSession, page: int = 1, page_size: int = 20
    ) -> ExpenseListResponse:
        """List expenses with pagination."""
        offset = (page - 1) * page_size
 
        count_result = await db.execute(select(func.count(Expense.id)))
        total = count_result.scalar()
 
        result = await db.execute(
            select(Expense)
            .order_by(desc(Expense.created_at))
            .offset(offset)
            .limit(page_size)
        )
        expenses = result.scalars().all()
 
        return ExpenseListResponse(
            total=total,
            page=page,
            page_size=page_size,
            expenses=[
                ExpenseSummary(
                    id=e.id,
                    file_name=e.file_name,
                    merchant_name=e.merchant_name,
                    total_amount=e.total_amount or 0,
                    primary_category=ExpenseCategory(e.primary_category) if e.primary_category else None,
                    ocr_confidence=e.ocr_confidence or 0,
                    created_at=e.created_at
                )
                for e in expenses
            ]
        )
 
    async def get_analytics(self, db: AsyncSession, days: int = 30) -> AnalyticsResponse:
        """Compute spending analytics over a period."""
        from datetime import timedelta
        from sqlalchemy import and_
        from collections import defaultdict
 
        since = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(Expense)
            .options(
                selectinload(Expense.items),
                selectinload(Expense.category_breakdowns)
            )
            .where(Expense.created_at >= since)
            .order_by(Expense.created_at)
        )
        expenses = result.scalars().all()
 
        if not expenses:
            return AnalyticsResponse(
                total_expenses=0,
                total_amount=0.0,
                avg_expense_amount=0.0,
                category_breakdown=[],
                top_merchants=[],
                spending_trend=[],
                period_days=days
            )
 
        total_amount = sum(e.total_amount or 0 for e in expenses)
        avg_amount = total_amount / len(expenses)

        cat_agg = defaultdict(lambda: {"total": 0.0, "count": 0})
        for expense in expenses:
            for cb in expense.category_breakdowns:
                cat_agg[cb.category]["total"] += cb.total_amount
                cat_agg[cb.category]["count"] += cb.item_count
 
        category_breakdown = [
            CategoryBreakdown(
                category=ExpenseCategory(cat),
                total_amount=round(data["total"], 2),
                item_count=data["count"],
                percentage=round(data["total"] / total_amount * 100, 2) if total_amount else 0,
            )
            for cat, data in cat_agg.items()
        ]
        category_breakdown.sort(key=lambda x: x.total_amount, reverse=True)

        merchant_agg = defaultdict(lambda: {"total": 0.0, "count": 0})
        for e in expenses:
            name = e.merchant_name or "Unknown"
            merchant_agg[name]["total"] += e.total_amount or 0
            merchant_agg[name]["count"] += 1
 
        top_merchants = sorted(
            [{"merchant": k, "total": round(v["total"], 2), "visits": v["count"]}
             for k, v in merchant_agg.items()],
            key=lambda x: x["total"], reverse=True
        )[:10]
 
        daily_agg = defaultdict(lambda: {"amount": 0.0, "count": 0})
        for e in expenses:
            day = e.created_at.strftime("%Y-%m-%d")
            daily_agg[day]["amount"] += e.total_amount or 0
            daily_agg[day]["count"] += 1
 
        spending_trend = [
            SpendingTrendPoint(
                date=day,
                amount=round(data["amount"], 2),
                count=data["count"]
            )
            for day, data in sorted(daily_agg.items())
        ]
 
        return AnalyticsResponse(
            total_expenses=len(expenses),
            total_amount=round(total_amount, 2),
            avg_expense_amount=round(avg_amount, 2),
            category_breakdown=category_breakdown,
            top_merchants=top_merchants,
            spending_trend=spending_trend,
            period_days=days
        )
 
    def _expense_to_response(self, expense: Expense) -> ExpenseUploadResponse:
        items = [
            LineItemResponse(
                id=item.id,
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                total_price=item.total_price,
                category=ExpenseCategory(item.category) if item.category else ExpenseCategory.OTHERS,
                category_confidence=item.category_confidence or 0,
            )
            for item in expense.items
        ]
        breakdown = [
            CategoryBreakdown(
                category=ExpenseCategory(cb.category),
                total_amount=cb.total_amount,
                item_count=cb.item_count,
                percentage=cb.percentage,
            )
            for cb in expense.category_breakdowns
        ]
        return ExpenseUploadResponse(
            expense_id=expense.id,
            status=expense.status,
            file_name=expense.file_name,
            ocr_confidence=expense.ocr_confidence or 0,
            confidence_level=self._get_confidence_level(expense.ocr_confidence or 0),
            merchant_name=expense.merchant_name,
            date=expense.receipt_date,
            payment_method=expense.payment_method,
            items=items,
            subtotal=expense.subtotal,
            tax=expense.tax_amount,
            discount=expense.discount_amount,
            total_amount=expense.total_amount or 0,
            category_breakdown=breakdown,
            primary_category=ExpenseCategory(expense.primary_category) if expense.primary_category else None,
            processing_time_ms=expense.processing_time_ms or 0,
            created_at=expense.created_at or datetime.utcnow()
        )
 
 
def get_expense_service() -> ExpenseProcessingService:
    return ExpenseProcessingService()