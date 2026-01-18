from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
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
    try:
        file_path = save_upload(file)
        # analyze_image returns None if the image cannot be read or is invalid
        analysis = analyze_image(file_path)

        if analysis is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

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
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error internally but return generic message to user
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")