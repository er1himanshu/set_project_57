from pydantic import BaseModel
from typing import Optional

class ImageResultSchema(BaseModel):
    id: int
    filename: str
    width: int
    height: int
    blur_score: float
    brightness_score: float
    contrast_score: float
    passed: bool
    reason: str
    
    # New ecommerce fields
    description: Optional[str] = None
    aspect_ratio: Optional[float] = None
    sharpness_score: Optional[float] = None
    background_score: Optional[float] = None
    has_watermark: Optional[bool] = False
    description_consistency: Optional[str] = None
    improvement_suggestions: Optional[str] = None

    class Config:
        from_attributes = True