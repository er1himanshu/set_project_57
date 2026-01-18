from sqlalchemy import Column, Integer, String, Float, Boolean, Text
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
    
    # New ecommerce fields
    description = Column(Text, nullable=True)
    aspect_ratio = Column(Float, nullable=True)
    sharpness_score = Column(Float, nullable=True)
    background_score = Column(Float, nullable=True)
    has_watermark = Column(Boolean, default=False)
    description_consistency = Column(String, nullable=True)
    improvement_suggestions = Column(Text, nullable=True)
    
    # CLIP fields
    clip_similarity_score = Column(Float, nullable=True)
    clip_mismatch = Column(Boolean, nullable=True)