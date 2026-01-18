from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import ImageResult
from ..schemas import ImageResultSchema
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


@router.get("/analyze/{image_id}", response_model=ImageResultSchema)
async def analyze_result(
    image_id: int,
    db: Session = Depends(get_db)
):
    """
    Get analysis result by image ID.
    
    Args:
        image_id: ID of the image result to retrieve
        db: Database session
        
    Returns:
        ImageResultSchema: Analysis result
        
    Raises:
        HTTPException: If result not found
    """
    try:
        result = db.query(ImageResult).filter(ImageResult.id == image_id).first()
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis result with ID {image_id} not found"
            )
        
        logger.info(f"Retrieved analysis result {image_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching analysis {image_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis result"
        )