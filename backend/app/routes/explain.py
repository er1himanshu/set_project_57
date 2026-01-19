"""
API endpoint for CLIP explainability - attention-based heatmap generation
"""

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import logging

from ..services.storage import save_upload
from ..services.explainability import generate_attention_heatmap
from ..services.mismatch_detector import MismatchDetectionUnavailableError
from ..config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS

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


@router.post("/explain")
async def explain_clip_similarity(
    file: UploadFile = File(...),
    description: str = Form(...)
):
    """
    Generate CLIP explainability heatmap showing which image regions contribute
    to the similarity score with the given description.
    
    This endpoint uses attention rollout from CLIP's Vision Transformer to create
    a heatmap overlay that visualizes the important regions for image-text matching.
    
    Args:
        file: Image file to analyze (max 10MB, jpg/png/gif/webp)
        description: Product description to compare against (min 10 characters)
        
    Returns:
        dict: Explanation results including:
            - heatmap_base64: Base64-encoded PNG image with heatmap overlay
            - similarity_score: CLIP similarity score (0-1)
            - message: Status message
            - filename: Original filename
            - description: The provided description
            
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
                detail="Description is required for explainability"
            )
        
        # Sanitize and validate description length
        description = description.strip()
        if len(description) < 10:
            raise HTTPException(
                status_code=400,
                detail="Description must be at least 10 characters long"
            )
        if len(description) > 500:
            logger.warning(f"Description truncated from {len(description)} to 500 characters")
            description = description[:500]  # Truncate to limit
        
        # Save file temporarily
        file_path, unique_filename = save_upload(file)
        logger.info(f"Generating explanation for file: {unique_filename}")
        
        # Generate attention heatmap
        heatmap_base64, similarity_score, message = generate_attention_heatmap(
            file_path,
            description
        )
        
        # Check if explanation generation succeeded
        if heatmap_base64 is None:
            logger.warning(f"Explainability unavailable: {message}")
            raise HTTPException(
                status_code=503,
                detail="CLIP explainability is currently unavailable. The AI model required for this feature is not accessible."
            )
        
        logger.info(f"Explanation generated: {message}")
        
        return {
            "filename": unique_filename,
            "description": description,
            "heatmap_base64": heatmap_base64,
            "similarity_score": similarity_score,
            "message": message
        }
        
    except HTTPException:
        # Clean up file on validation errors
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise
    except MismatchDetectionUnavailableError as e:
        # CLIP model unavailable - return specific error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        
        logger.warning(f"Explainability unavailable for file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="CLIP explainability is currently unavailable. The AI model required for this feature is not accessible."
        )
    except FileNotFoundError as e:
        # File not found error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        
        logger.error(f"Image file error for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Image file could not be processed."
        )
    except Exception as e:
        # Clean up file on any error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")
        
        logger.error(f"Explainability error for file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during explanation generation. Please try again."
        )
    finally:
        # Always clean up the temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file {file_path}: {str(cleanup_error)}")
