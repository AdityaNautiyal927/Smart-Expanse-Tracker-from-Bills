from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from app.api.routes import expenses, health, analytics
from app.db.database import init_db
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Smart Expense Tracker...")
    await init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info("🛑 Shutting down Smart Expense Tracker...")


app = FastAPI(
    title="Smart Expense Tracker",
    description="""
    ## Smart Bill & Receipt OCR Expense Tracker

    Upload receipt/bill images and automatically extract structured expense data.

    ### Features
    - 📸 **OCR Processing**: Extract text from receipt images using EasyOCR
    - 🧠 **Smart Parsing**: Identify items and prices using regex + heuristics
    - 🏷️ **Auto-Categorization**: Rule-based + ML categorization engine
    - 📊 **Analytics**: Category-wise breakdowns and spending insights
    - 💾 **Persistent Storage**: SQLite (dev) → PostgreSQL (prod) ready

    ### Architecture
    Modular monolith → Microservices ready
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(expenses.router, prefix="/api/v1/expenses", tags=["Expenses"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Smart Expense Tracker API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }