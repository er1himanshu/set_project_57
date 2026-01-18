"""
API endpoint for image-text mismatch detection
"""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import logging

from ..services.storage import save_upload
from ..services.mismatch_detector import check_image_text_similarity
from ..config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MISMATCH_THRESHOLD

router = APIRouter()
logger = logging.getLogger(__name__)


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
    
    # Check file size
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


@router.post("/check-mismatch")
async def check_mismatch(
    file: UploadFile = File(...),
    description: str = Form(...),
    threshold: Optional[float] = Form(None)
):
    """
    Check for image-text mismatch using CLIP model.
    
    Args:
        file: Image file to analyze (max 10MB, jpg/png/gif/webp)
        description: Product description to compare against the image
        threshold: Optional custom similarity threshold (0-1, lower = stricter)
        
    Returns:
        dict: Mismatch detection results including similarity score and status
        
    Raises:
        HTTPException: If validation or processing fails
    """
    file_path = None
    try:
        # Validate file
        validate_file(file)
        
        # Validate description
        if not description or description.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Description is required for mismatch detection"
            )
        
        # Sanitize and validate description length
        description = description.strip()
        if len(description) < 10:
            raise HTTPException(
                status_code=400,
                detail="Description must be at least 10 characters long"
            )
        if len(description) > 500:
            description = description[:500]  # Truncate to limit
        
        # Validate threshold if provided
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Threshold must be between 0 and 1"
                )
        
        # Save file temporarily
        file_path, unique_filename = save_upload(file)
        logger.info(f"Checking mismatch for file: {unique_filename}")
        
        # Run mismatch detection
        result = check_image_text_similarity(file_path, description, threshold)
        
        logger.info(f"Mismatch check complete: {result['message']}")
        
        return {
            "filename": unique_filename,
            "description": description,
            "has_mismatch": result["has_mismatch"],
            "similarity_score": result["similarity_score"],
            "threshold": result["threshold"],
            "message": result["message"],
            "recommendation": (
                "Image and description appear to mismatch. Consider reviewing." 
                if result["has_mismatch"] 
                else "Image and description match well."
            )
        }
        
    except HTTPException:
        # Clean up file on validation errors
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise
    except Exception as e:
        # Clean up file on any error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")
        
        logger.error(f"Mismatch detection error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during mismatch detection. Please try again."
        )
    finally:
        # Always clean up the temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")
