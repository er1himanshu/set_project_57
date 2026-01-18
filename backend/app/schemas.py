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
    
    # CLIP fields
    clip_similarity_score: Optional[float] = None
    clip_mismatch: Optional[bool] = None

    class Config:
        from_attributes = True


class CLIPCheckRequest(BaseModel):
    """Request schema for CLIP mismatch detection."""
    image_path: Optional[str] = None
    description: str
    threshold: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_path": "/path/to/image.jpg",
                "description": "Red leather handbag with gold hardware",
                "threshold": 0.25
            }
        }


class CLIPCheckResponse(BaseModel):
    """Response schema for CLIP mismatch detection."""
    is_match: Optional[bool]
    similarity_score: Optional[float]
    decision: str
    threshold_used: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_match": True,
                "similarity_score": 0.87,
                "decision": "Match (score: 0.870 >= 0.250)",
                "threshold_used": 0.25
            }
        }