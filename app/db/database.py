from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings
import logging
 
logger = logging.getLogger(__name__)
 
 
class Base(DeclarativeBase):
    pass
 
 
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,

    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)
 

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)
 
 
async def init_db():
    from app.models import expense  
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")
 
 
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()