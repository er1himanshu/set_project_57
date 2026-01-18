from fastapi import APIRouter, Depends
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