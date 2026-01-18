"""
API routes for CLIP-based image-text similarity analysis.
"""

import os
import logging
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..services.clip_analyzer import get_clip_analyzer, CLIPAnalyzer
from ..services.storage import save_upload
from ..config import CLIP_SIMILARITY_THRESHOLD, DEFAULT_CATEGORIES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/clip", tags=["CLIP"])


# Pydantic schemas for request/response
class CLIPAnalysisRequest(BaseModel):
    """Request schema for CLIP analysis with image path and text."""
    image_path: str = Field(..., description="Path to image file")
    text: str = Field(..., description="Text description to compare")
    threshold: Optional[float] = Field(
        None, 
        description="Similarity threshold (default from config)",
        ge=0.0,
        le=1.0
    )


class CLIPAnalysisResponse(BaseModel):
    """Response schema for CLIP mismatch detection."""
    is_mismatch: bool = Field(..., description="Whether a mismatch was detected")
    similarity_score: float = Field(..., description="Cosine similarity score (0-1)")
    threshold_used: float = Field(..., description="Threshold used for detection")
    confidence: str = Field(..., description="Confidence level: high, medium, or low")
    match_quality: str = Field(..., description="Qualitative match description")


class CategoryMatch(BaseModel):
    """Schema for category match result."""
    category: str = Field(..., description="Category name")
    score: float = Field(..., description="Match score")


class CategoryComparisonResponse(BaseModel):
    """Response schema for category-based comparison."""
    top_matches: List[CategoryMatch] = Field(..., description="Top matching categories")


class BatchAnalysisRequest(BaseModel):
    """Request schema for batch analysis."""
    items: List[CLIPAnalysisRequest] = Field(..., description="List of image-text pairs to analyze")


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/analyze", response_model=CLIPAnalysisResponse)
async def analyze_image_text_similarity(
    file: UploadFile = File(...),
    text: str = Form(...),
    threshold: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Analyze image-text similarity using CLIP and detect mismatches.
    
    This endpoint accepts an image file and text description, computes their
    semantic similarity using CLIP, and returns whether they match or mismatch.
    
    Args:
        file: Image file to analyze
        text: Text description to compare against the image
        threshold: Optional custom threshold (default: 0.25)
        
    Returns:
        CLIPAnalysisResponse with mismatch detection results
        
    Raises:
        HTTPException: If analysis fails or file is invalid
    """
    file_path = None
    try:
        # Validate input
        if not text or text.strip() == "":
            raise HTTPException(status_code=400, detail="Text description is required")
        
        # Save uploaded file
        file_path, _ = save_upload(file)
        logger.info(f"Analyzing image-text similarity for: {file.filename}")
        
        # Get CLIP analyzer
        analyzer = get_clip_analyzer()
        
        # Detect mismatch
        result = analyzer.detect_mismatch(file_path, text, threshold)
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(
            f"Analysis complete. Similarity: {result['similarity_score']:.3f}, "
            f"Mismatch: {result['is_mismatch']}"
        )
        
        return CLIPAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file: {cleanup_error}")
        
        logger.error(f"CLIP analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during CLIP analysis: {str(e)}"
        )


@router.post("/analyze-batch")
async def analyze_batch(
    request: BatchAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Batch analyze multiple image-text pairs for efficiency.
    
    Note: This endpoint expects image_path to be a server-side path,
    not an upload. For uploaded files, use /analyze endpoint multiple times.
    
    Args:
        request: Batch request with list of image paths and texts
        
    Returns:
        List of CLIPAnalysisResponse objects
        
    Raises:
        HTTPException: If batch analysis fails
    """
    try:
        # Get CLIP analyzer
        analyzer = get_clip_analyzer()
        
        # Extract paths and texts
        image_paths = [item.image_path for item in request.items]
        texts = [item.text for item in request.items]
        
        # Batch analyze
        results = analyzer.batch_analyze(image_paths, texts)
        
        logger.info(f"Batch analysis complete for {len(results)} items")
        
        # Separate successful results from errors
        successful_results = []
        error_results = []
        
        for i, r in enumerate(results):
            if 'error' in r:
                error_results.append({
                    "index": i,
                    "error": r['error'],
                    "image_path": image_paths[i]
                })
            else:
                # Only include fields that are part of CLIPAnalysisResponse schema
                successful_results.append(CLIPAnalysisResponse(
                    is_mismatch=r['is_mismatch'],
                    similarity_score=r['similarity_score'],
                    threshold_used=r['threshold_used'],
                    confidence=r['confidence'],
                    match_quality=r['match_quality']
                ))
        
        return {
            "results": successful_results,
            "errors": error_results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch analysis: {str(e)}"
        )


@router.post("/classify", response_model=CategoryComparisonResponse)
async def classify_image_by_category(
    file: UploadFile = File(...),
    categories: Optional[str] = Form(None),
    top_k: int = Form(5),
    db: Session = Depends(get_db)
):
    """
    Classify image against product categories using zero-shot CLIP.
    
    This endpoint compares an image against a list of product categories
    and returns the top matching categories with scores.
    
    Args:
        file: Image file to classify
        categories: Optional comma-separated list of categories. 
                   If not provided, uses default categories.
        top_k: Number of top categories to return (default: 5)
        
    Returns:
        CategoryComparisonResponse with top category matches
        
    Raises:
        HTTPException: If classification fails
    """
    file_path = None
    try:
        # Save uploaded file
        file_path, _ = save_upload(file)
        logger.info(f"Classifying image: {file.filename}")
        
        # Parse categories
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(',') if c.strip()]
        
        # Get CLIP analyzer
        analyzer = get_clip_analyzer()
        
        # Perform classification
        results = analyzer.class_based_comparison(
            file_path,
            categories=category_list,
            top_k=top_k
        )
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Classification complete. Top category: {results[0]['category']}")
        
        return CategoryComparisonResponse(
            top_matches=[CategoryMatch(**r) for r in results]
        )
        
    except Exception as e:
        # Clean up file on error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file: {cleanup_error}")
        
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during classification: {str(e)}"
        )


@router.get("/categories")
async def get_supported_categories():
    """
    Get list of default supported product categories.
    
    Returns:
        dict: List of default categories available for classification
    """
    return {
        "categories": DEFAULT_CATEGORIES,
        "count": len(DEFAULT_CATEGORIES)
    }


@router.get("/config")
async def get_clip_config():
    """
    Get current CLIP configuration settings.
    
    Returns:
        dict: Current CLIP configuration
    """
    return {
        "similarity_threshold": CLIP_SIMILARITY_THRESHOLD,
        "default_categories_count": len(DEFAULT_CATEGORIES),
        "model_loaded": get_clip_analyzer() is not None
    }


@router.get("/health")
async def clip_health_check():
    """
    Health check endpoint for CLIP service.
    
    Returns:
        dict: Health status and model information
    """
    try:
        # Try to get analyzer to verify model loads
        analyzer = get_clip_analyzer()
        return {
            "status": "healthy",
            "service": "CLIP analyzer",
            "model_loaded": True,
            "device": analyzer.device
        }
    except Exception as e:
        logger.error(f"CLIP health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "CLIP analyzer",
            "model_loaded": False,
            "error": str(e)
        }
