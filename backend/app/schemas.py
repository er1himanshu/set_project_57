from pydantic import BaseModel

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

    class Config:
        from_attributes = True