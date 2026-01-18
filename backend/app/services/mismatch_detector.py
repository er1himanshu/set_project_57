"""
Image-Text Mismatch Detection using CLIP (Contrastive Language-Image Pre-training)

This module provides functionality to detect mismatches between product images and 
their descriptions using a pretrained multimodal CLIP model.
"""

import logging
from typing import Tuple, Optional
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ..config import MISMATCH_THRESHOLD, CLIP_MODEL_NAME

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request
_model_cache = {}


def get_clip_model():
    """
    Load and cache the CLIP model and processor.
    
    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The loaded model and processor
    """
    if 'model' not in _model_cache:
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        try:
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            _model_cache['model'] = model
            _model_cache['processor'] = processor
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}", exc_info=True)
            raise
    
    return _model_cache['model'], _model_cache['processor']


def detect_mismatch(image_path: str, description: str, threshold: Optional[float] = None) -> Tuple[bool, float, str]:
    """
    Detect if there's a mismatch between an image and its description using CLIP.
    
    Args:
        image_path: Path to the image file
        description: Text description of the product
        threshold: Optional custom threshold for mismatch detection
        
    Returns:
        Tuple[bool, float, str]: 
            - is_mismatch: True if mismatch detected
            - similarity_score: Similarity score between 0 and 1
            - message: Human-readable message about the result
    """
    # Handle empty or missing description
    if not description or description.strip() == "":
        return False, 1.0, "No description provided"
    
    # Use default threshold if not provided
    if threshold is None:
        threshold = MISMATCH_THRESHOLD
    
    try:
        # Load model and processor
        model, processor = get_clip_model()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(
            text=[description], 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Calculate similarity score
        # CLIP returns logits - normalize to 0-1 range for interpretability
        logits_per_image = outputs.logits_per_image
        # Normalize the single logit value to 0-1 range using sigmoid
        similarity_score = torch.sigmoid(logits_per_image / 100.0).item()
        
        # Determine if there's a mismatch using the provided threshold
        is_mismatch = similarity_score < threshold
        
        # Generate message
        if is_mismatch:
            message = f"Mismatch detected (score: {similarity_score:.2f})"
        else:
            message = f"Match confirmed (score: {similarity_score:.2f})"
        
        logger.info(f"Mismatch detection: {message}")
        
        return is_mismatch, float(similarity_score), message
        
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return False, 0.0, "Image file not found"
    except Exception as e:
        logger.error(f"Error in mismatch detection: {str(e)}", exc_info=True)
        return False, 0.0, f"Error: {str(e)}"


def check_image_text_similarity(image_path: str, description: str, threshold: Optional[float] = None) -> dict:
    """
    Check similarity between image and text description with detailed results.
    
    Args:
        image_path: Path to the image file
        description: Text description of the product
        threshold: Optional custom threshold (uses config default if None)
        
    Returns:
        dict: Detailed results including mismatch status, score, and message
    """
    if threshold is None:
        threshold = MISMATCH_THRESHOLD
    
    is_mismatch, similarity_score, message = detect_mismatch(image_path, description, threshold)
    
    return {
        "has_mismatch": is_mismatch,
        "similarity_score": similarity_score,
        "threshold": threshold,
        "message": message,
        "description_provided": bool(description and description.strip())
    }
