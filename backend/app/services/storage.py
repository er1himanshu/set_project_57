import os
import uuid
from pathlib import Path
from fastapi import UploadFile
from ..config import UPLOAD_DIR
import logging

logger = logging.getLogger(__name__)


def save_upload(file: UploadFile) -> str:
    """
    Save uploaded file to disk with a unique filename.
    
    Args:
        file: Uploaded file object
        
    Returns:
        str: Full path to saved file
        
    Raises:
        Exception: If file save fails
    """
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate unique filename to prevent collisions
        file_ext = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Write file to disk
        with open(file_path, "wb") as f:
            content = file.file.read()
            f.write(content)
        
        logger.info(f"File saved: {unique_filename}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}", exc_info=True)
        raise