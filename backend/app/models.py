from sqlalchemy import Column, Integer, String, Float, Boolean
from .database import Base

class ImageResult(Base):
    __tablename__ = "image_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    width = Column(Integer)
    height = Column(Integer)
    blur_score = Column(Float)
    brightness_score = Column(Float)
    contrast_score = Column(Float)
    passed = Column(Boolean, default=False)
    reason = Column(String)