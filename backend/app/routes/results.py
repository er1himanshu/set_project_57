from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import ImageResult
from ..schemas import ImageResultSchema
from typing import List

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/results", response_model=List[ImageResultSchema])
def get_all_results(db: Session = Depends(get_db)):
    return db.query(ImageResult).all()

@router.get("/results/{result_id}", response_model=ImageResultSchema)
def get_result_detail(result_id: int, db: Session = Depends(get_db)):
    result = db.query(ImageResult).filter(ImageResult.id == result_id).first()
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result