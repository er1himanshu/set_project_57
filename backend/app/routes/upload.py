import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..services.storage import save_upload
from ..services.image_quality import analyze_image
from ..models import ImageResult
from ..config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file for size and type.
    
    Args:
        file: Uploaded file object
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (read file to get actual size)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...), 
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload and analyze a product image.
    
    Args:
        file: Image file to upload (max 10MB, jpg/png/gif/webp)
        description: Optional product description for consistency checking
        db: Database session
        
    Returns:
        dict: Upload confirmation with result_id
        
    Raises:
        HTTPException: If upload or analysis fails
    """
    try:
        # Validate file
        validate_file(file)
        
        # Sanitize description
        if description:
            description = description.strip()[:500]  # Limit description length
        
        # Save file
        file_path = save_upload(file)
        logger.info(f"File uploaded: {file.filename}")
        
        # Analyze image
        analysis = analyze_image(file_path, description)

        if analysis is None:
            # Clean up the uploaded file if analysis fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file. Please upload a valid image."
            )

        # Store result in database
        result = ImageResult(
            filename=file.filename,
            width=analysis["width"],
            height=analysis["height"],
            blur_score=analysis["blur_score"],
            brightness_score=analysis["brightness_score"],
            contrast_score=analysis["contrast_score"],
            passed=analysis["passed"],
            reason=analysis["reason"],
            description=description,
            aspect_ratio=analysis.get("aspect_ratio"),
            sharpness_score=analysis.get("sharpness_score"),
            background_score=analysis.get("background_score"),
            has_watermark=analysis.get("has_watermark", False),
            description_consistency=analysis.get("description_consistency"),
            improvement_suggestions=analysis.get("improvement_suggestions")
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        
        logger.info(f"Analysis complete for {file.filename}, result_id: {result.id}")

        return {
            "message": "Image uploaded and analyzed successfully",
            "result_id": result.id,
            "passed": result.passed
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the error internally but return generic message to user
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during upload. Please try again."
        )