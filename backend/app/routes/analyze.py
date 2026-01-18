from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import ImageResult
from ..schemas import ImageResultSchema

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/analyze/{image_id}", response_model=ImageResultSchema)
def analyze_result(image_id: int, db: Session = Depends(get_db)):
    result = db.query(ImageResult).filter(ImageResult.id == image_id).first()
    return result