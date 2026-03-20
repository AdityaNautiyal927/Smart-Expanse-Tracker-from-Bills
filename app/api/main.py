from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
 
from app.api.routes import expenses, health, analytics
from app.db.database import init_db
from app.core.config import settings