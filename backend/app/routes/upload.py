from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..services.storage import save_upload
from ..services.image_quality import analyze_image
from ..models import ImageResult

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload")
def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = save_upload(file)
    analysis = analyze_image(file_path)

    if analysis is None:
        return {"error": "Invalid image"}

    result = ImageResult(
        filename=file.filename,
        width=analysis["width"],
        height=analysis["height"],
        blur_score=analysis["blur_score"],
        brightness_score=analysis["brightness_score"],
        contrast_score=analysis["contrast_score"],
        passed=analysis["passed"],
        reason=analysis["reason"]
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    return {"message": "Uploaded & analyzed", "result_id": result.id}