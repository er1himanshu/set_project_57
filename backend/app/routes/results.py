from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import ImageResult
from ..schemas import ImageResultSchema
from typing import List, Optional
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


@router.get("/results", response_model=List[ImageResultSchema])
async def get_all_results(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    passed: Optional[bool] = Query(None, description="Filter by pass/fail status"),
    db: Session = Depends(get_db)
):
    """
    Get all analysis results with optional filtering and pagination.
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        passed: Optional filter for passed/failed images
        db: Database session
        
    Returns:
        List[ImageResultSchema]: List of image analysis results
    """
    try:
        query = db.query(ImageResult)
        
        # Apply filters
        if passed is not None:
            query = query.filter(ImageResult.passed == passed)
        
        # Order by most recent first
        query = query.order_by(ImageResult.id.desc())
        
        # Apply pagination
        results = query.offset(skip).limit(limit).all()
        
        logger.info(f"Retrieved {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve results"
        )


@router.get("/results/{result_id}", response_model=ImageResultSchema)
async def get_result_detail(
    result_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information for a specific analysis result.
    
    Args:
        result_id: ID of the result to retrieve
        db: Database session
        
    Returns:
        ImageResultSchema: Detailed analysis result
        
    Raises:
        HTTPException: If result not found
    """
    try:
        result = db.query(ImageResult).filter(ImageResult.id == result_id).first()
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Result with ID {result_id} not found"
            )
        
        logger.info(f"Retrieved result {result_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching result {result_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve result"
        )