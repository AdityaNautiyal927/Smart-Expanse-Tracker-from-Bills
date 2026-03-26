from app.core.config import settings

try:
    from celery import Celery

    celery_app = Celery(
        "expense_tracker",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=["app.tasks.celery_tasks"],
    )

    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_routes={
            "app.tasks.celery_tasks.run_ocr": {"queue": "ocr"},
            "app.tasks.celery_tasks.run_parse": {"queue": "parse"},
            "app.tasks.celery_tasks.run_categorize": {"queue": "categorize"},
            "app.tasks.celery_tasks.process_bill_async": {"queue": "pipeline"},
        },
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_annotations={
            "app.tasks.celery_tasks.run_ocr": {"rate_limit": "10/m"}
        },
    )

except ImportError:
    celery_app = None


if celery_app:

    @celery_app.task(
        name="app.tasks.celery_tasks.run_ocr",
        bind=True,
        max_retries=3,
        default_retry_delay=5,
    )
    def run_ocr(self, image_path: str) -> dict:

        try:
            from app.services.ocr.ocr_service import get_ocr_service
            service = get_ocr_service()
            result = service.process(image_path)
            return result.model_dump()
        except Exception as exc:
            raise self.retry(exc=exc)

    @celery_app.task(name="app.tasks.celery_tasks.run_parse")
    def run_parse(ocr_result: dict) -> dict:
        from app.services.parser.parser_service import get_parser_service
        service = get_parser_service()
        result = service.process(ocr_result["raw_text"])
        return result.model_dump()

    @celery_app.task(name="app.tasks.celery_tasks.run_categorize")
    def run_categorize(parsed_receipt: dict) -> dict:
        from app.services.categorizer.categorization_service import get_categorization_service
        from app.schemas.expense import LineItemBase, ParsedReceipt

        service = get_categorization_service()
        receipt = ParsedReceipt(**parsed_receipt)
        categorized = service.categorize_items(receipt.items)
        breakdown = service.compute_breakdown(categorized, receipt.total)

        return {
            "items": [i.model_dump() for i in categorized],
            "breakdown": [b.model_dump() for b in breakdown],
        }

    @celery_app.task(
        name="app.tasks.celery_tasks.process_bill_async",
        bind=True,
    )
    def process_bill_async(self, file_path: str, file_name: str, expense_id: int):
        import asyncio
        from app.services.ocr.ocr_service import get_ocr_service
        from app.services.parser.parser_service import get_parser_service
        from app.services.categorizer.categorization_service import get_categorization_service

        ocr_result = get_ocr_service().process(file_path)
        parsed = get_parser_service().process(ocr_result.raw_text)
        categorizer = get_categorization_service()
        items = categorizer.categorize_items(parsed.items)
        breakdown = categorizer.compute_breakdown(items, parsed.total)

        return {
            "expense_id": expense_id,
            "status": "completed",
            "total": parsed.total,
            "item_count": len(items),
            "categories": [b.category.value for b in breakdown],
        }