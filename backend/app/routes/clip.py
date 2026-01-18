"""
CLIP mismatch detection API endpoints.

Provides REST API endpoints for checking image-text similarity
and detecting mismatches using CLIP models.
"""

import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from ..schemas import CLIPCheckRequest, CLIPCheckResponse
from ..services.clip_service import check_mismatch_clip
from ..services.storage import save_upload
from pathlib import Path

router = APIRouter(prefix="/clip", tags=["CLIP"])
logger = logging.getLogger(__name__)


@router.post("/check-mismatch", response_model=CLIPCheckResponse)
async def check_mismatch(
    file: UploadFile = File(...),
    description: str = Form(...),
    threshold: Optional[float] = Form(None)
):
    """
    Check for image-description mismatch using CLIP.
    
    This endpoint accepts an image and text description, and returns:
    - Similarity score (0-1)
    - Match/mismatch decision
    - Threshold used for decision
    
    Args:
        file: Image file to analyze
        description: Text description to compare with image
        threshold: Optional custom threshold (default from config)
        
    Returns:
        CLIPCheckResponse with similarity score and match decision
        
    Raises:
        HTTPException: If analysis fails
        
    Example:
        ```python
        import requests
        
        files = {"file": open("product.jpg", "rb")}
        data = {
            "description": "Red leather handbag",
            "threshold": 0.3
        }
        response = requests.post(
            "http://localhost:8000/clip/check-mismatch",
            files=files,
            data=data
        )
        print(response.json())
        # Output: {
        #   "is_match": true,
        #   "similarity_score": 0.87,
        #   "decision": "Match (score: 0.870 >= 0.300)",
        #   "threshold_used": 0.3
        # }
        ```
    """
    file_path = None
    try:
        # Validate inputs
        if not description or description.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Description is required"
            )
        
        # Save uploaded file temporarily
        file_path, unique_filename = save_upload(file)
        logger.info(f"File uploaded for CLIP check: {unique_filename}")
        
        # Perform CLIP mismatch detection
        result = check_mismatch_clip(
            image_path=file_path,
            description=description.strip(),
            threshold=threshold
        )
        
        logger.info(f"CLIP check result: {result['decision']}")
        
        return CLIPCheckResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in CLIP mismatch check: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CLIP mismatch check: {str(e)}"
        )
    finally:
        # Clean up uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up file {file_path}: {str(cleanup_error)}")


@router.post("/check-mismatch-by-path", response_model=CLIPCheckResponse)
async def check_mismatch_by_path(request: CLIPCheckRequest):
    """
    Check for image-description mismatch using CLIP (for existing images).
    
    This endpoint is useful for checking already-uploaded images without
    re-uploading them.
    
    Args:
        request: CLIPCheckRequest with image_path, description, and optional threshold
        
    Returns:
        CLIPCheckResponse with similarity score and match decision
        
    Raises:
        HTTPException: If analysis fails or image not found
        
    Example:
        ```python
        import requests
        
        data = {
            "image_path": "/path/to/uploaded/image.jpg",
            "description": "Blue cotton t-shirt",
            "threshold": 0.25
        }
        response = requests.post(
            "http://localhost:8000/clip/check-mismatch-by-path",
            json=data
        )
        print(response.json())
        ```
    """
    try:
        # Validate inputs
        if not request.description or request.description.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Description is required"
            )
        
        if not request.image_path:
            raise HTTPException(
                status_code=400,
                detail="Image path is required"
            )
        
        # Check if file exists
        if not os.path.exists(request.image_path):
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {request.image_path}"
            )
        
        # Perform CLIP mismatch detection
        result = check_mismatch_clip(
            image_path=request.image_path,
            description=request.description.strip(),
            threshold=request.threshold
        )
        
        logger.info(f"CLIP check result for {request.image_path}: {result['decision']}")
        
        return CLIPCheckResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in CLIP mismatch check: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CLIP mismatch check: {str(e)}"
        )
