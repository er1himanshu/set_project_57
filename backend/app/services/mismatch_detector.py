"""
Image-Text Mismatch Detection using CLIP (Contrastive Language-Image Pre-training)

This module provides functionality to detect mismatches between product images and 
their descriptions using a pretrained multimodal CLIP model.
"""

import logging
import re
from typing import Tuple, Optional
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ..config import MISMATCH_THRESHOLD, CLIP_MODEL_NAME, CATEGORY_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request
_model_cache = {}

# Error message constant
MISMATCH_DETECTION_UNAVAILABLE_MSG = "Image-text mismatch detection is unavailable"

# Product categories for category-based mismatch detection
PRODUCT_CATEGORIES = [
    "shoes", "sneakers", "boots", "sandals",
    "clothing", "shirt", "pants", "dress", "jacket", "coat",
    "bag", "handbag", "backpack", "purse", "wallet",
    "watch", "jewelry", "necklace", "bracelet", "ring",
    "electronics", "phone", "laptop", "computer", "tablet",
    "furniture", "chair", "table", "sofa", "bed",
    "toys", "doll", "action figure", "game",
    "bicycle", "bike", "motorcycle", "scooter",
    "sports equipment", "ball", "bat", "racket",
    "accessories", "hat", "sunglasses", "belt", "scarf",
    "beauty products", "makeup", "cosmetics", "perfume",
    "food", "beverage", "drink",
    "books", "magazines",
    "kitchen items", "cookware", "appliances",
    "home decor", "artwork", "vase", "lamp"
]


class MismatchDetectionUnavailableError(Exception):
    """Raised when mismatch detection is not available (e.g., model not loaded)."""
    pass


def get_clip_model():
    """
    Load and cache the CLIP model and processor.
    
    Returns:
        Tuple[CLIPModel, CLIPProcessor]: The loaded model and processor
        
    Raises:
        MismatchDetectionUnavailableError: If model cannot be loaded
    """
    if 'model' not in _model_cache:
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        try:
            # Try to load from local cache only (fail fast if not available)
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
            _model_cache['model'] = model
            _model_cache['processor'] = processor
            logger.info("CLIP model loaded successfully from local cache")
        except Exception as e:
            logger.info(f"CLIP model not available locally: {type(e).__name__}")
            # Mark model as unavailable - don't retry downloading
            _model_cache['model'] = None
            _model_cache['processor'] = None
            raise MismatchDetectionUnavailableError(
                "CLIP model is not available locally. Image-text mismatch detection will be skipped."
            ) from e
    
    # Check if model loading previously failed
    if _model_cache.get('model') is None:
        raise MismatchDetectionUnavailableError(
            "CLIP model is not available. Image-text mismatch detection will be skipped."
        )
    
    return _model_cache['model'], _model_cache['processor']


def detect_image_category(image_path: str, top_k: int = 3) -> Tuple[str, float]:
    """
    Detect the most likely product category for an image using CLIP.
    
    Args:
        image_path: Path to the image file
        top_k: Number of top categories to consider
        
    Returns:
        Tuple[str, float]: Best matching category and its confidence score
        
    Raises:
        MismatchDetectionUnavailableError: If CLIP model is not available
    """
    try:
        # Load model and processor
        model, processor = get_clip_model()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs with all product categories
        inputs = processor(
            text=PRODUCT_CATEGORIES,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate probabilities for each category
        logits_per_image = outputs.logits_per_image
        probs = torch.softmax(logits_per_image, dim=1)[0]
        
        # Get top-k categories with highest probabilities
        top_probs, top_indices = torch.topk(probs, min(top_k, len(PRODUCT_CATEGORIES)))
        
        # Return the best category
        best_category = PRODUCT_CATEGORIES[top_indices[0].item()]
        best_score = top_probs[0].item()
        
        logger.info(f"Image category detected: {best_category} (confidence: {best_score:.2f})")
        
        return best_category, float(best_score)
        
    except MismatchDetectionUnavailableError:
        raise
    except Exception as e:
        logger.warning(f"Category detection failed: {type(e).__name__}: {str(e)}")
        raise MismatchDetectionUnavailableError(
            "Category detection unavailable"
        ) from e


def extract_category_from_description(description: str) -> Optional[str]:
    """
    Extract the most likely product category from a description using word boundary matching.
    
    Args:
        description: Product description text
        
    Returns:
        str or None: Detected category from description, or None if no match
    """
    if not description:
        return None
    
    description_lower = description.lower()
    
    # Check for exact category matches using word boundaries
    for category in PRODUCT_CATEGORIES:
        # Use regex with word boundaries to avoid partial word matches
        pattern = r'\b' + re.escape(category) + r'\b'
        if re.search(pattern, description_lower):
            return category
    
    return None


def detect_mismatch(image_path: str, description: str, threshold: Optional[float] = None) -> Tuple[bool, Optional[float], str]:
    """
    Detect if there's a mismatch between an image and its description using CLIP.
    
    Args:
        image_path: Path to the image file
        description: Text description of the product
        threshold: Optional custom threshold for mismatch detection
        
    Returns:
        Tuple[bool, Optional[float], str]: 
            - is_mismatch: True if mismatch detected, False otherwise
            - similarity_score: Similarity score between 0 and 1, or None if detection unavailable
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
        
        # Try to detect category-based mismatch for more informative messaging
        category_message = None
        try:
            # Detect the category from the image
            image_category, category_confidence = detect_image_category(image_path, top_k=3)
            
            # Extract category from description
            description_category = extract_category_from_description(description)
            
            # Check if there's a category conflict
            if description_category and image_category:
                # Check if they are different categories and confidence is high
                if description_category != image_category and category_confidence > CATEGORY_CONFIDENCE_THRESHOLD:
                    category_message = f"Description mentions '{description_category}', but image looks like '{image_category}'"
                    logger.info(f"Category mismatch detected: {category_message}")
        except Exception as e:
            # Category detection is optional - don't fail if it doesn't work
            logger.debug(f"Category detection failed: {str(e)}")
        
        # Generate message
        if is_mismatch:
            if category_message:
                message = f"Mismatch detected (score: {similarity_score:.2f}). {category_message}."
            else:
                message = f"Mismatch detected (score: {similarity_score:.2f})"
        else:
            message = f"Match confirmed (score: {similarity_score:.2f})"
        
        logger.info(f"Mismatch detection: {message}")
        
        return is_mismatch, float(similarity_score), message
        
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}") from e
    except MismatchDetectionUnavailableError:
        # CLIP model is unavailable - return gracefully without failing
        logger.warning("Mismatch detection unavailable: CLIP model not loaded")
        return False, None, MISMATCH_DETECTION_UNAVAILABLE_MSG
    except Exception as e:
        # For other errors, log and return unavailable status
        logger.warning(f"Mismatch detection failed: {type(e).__name__}: {str(e)}")
        return False, None, MISMATCH_DETECTION_UNAVAILABLE_MSG


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
        "similarity_score": similarity_score,  # Can be None if detection unavailable
        "threshold": threshold,
        "message": message,
        "description_provided": bool(description and description.strip())
    }
