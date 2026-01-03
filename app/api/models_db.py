# Pydantic définit la forme des données qui entrent/sortent,
# SQLAlchemy définit la forme des données qui dorment en base.

# imports
from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.database import Base

# ============== Tables ========================


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String(12), primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    inputs = Column(JSONB)

    prediction = Column(Integer)
    confidence = Column(Float)
    class_name = Column(String)

    model_version = Column(String, default="v1.0.0")
