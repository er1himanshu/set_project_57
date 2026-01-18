"""
CLIP-based image-text similarity service for mismatch detection.

This module provides functionality to:
1. Load CLIP models (pre-trained or fine-tuned)
2. Compute similarity scores between images and text descriptions
3. Detect mismatches based on configurable thresholds
"""

import os
import torch
import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, Optional
from ..config import (
    CLIP_MODEL_NAME,
    CLIP_FINE_TUNED_MODEL_PATH,
    CLIP_SIMILARITY_THRESHOLD,
    CLIP_DEVICE,
    CLIP_CACHE_DIR
)

logger = logging.getLogger(__name__)


class CLIPMismatchDetector:
    """
    CLIP-based mismatch detector for image-text similarity.
    
    This class handles:
    - Loading pre-trained or fine-tuned CLIP models
    - Computing similarity scores between images and text
    - Determining match/mismatch based on threshold
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize CLIP mismatch detector.
        
        Args:
            model_path: Path to fine-tuned model, or None to use default model
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.device = device or CLIP_DEVICE
        
        # Auto-detect CUDA availability if not specified
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Determine which model to load
        self.model_path = model_path or CLIP_FINE_TUNED_MODEL_PATH or CLIP_MODEL_NAME
        
        # Create cache directory if it doesn't exist
        os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and processor."""
        try:
            logger.info(f"Loading CLIP model from: {self.model_path}")
            
            # Load processor and model
            if os.path.isdir(self.model_path):
                # Load from local fine-tuned model
                self.processor = CLIPProcessor.from_pretrained(self.model_path)
                self.model = CLIPModel.from_pretrained(self.model_path)
                logger.info("Loaded fine-tuned CLIP model from local path")
            else:
                # Load from HuggingFace hub
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_path,
                    cache_dir=CLIP_CACHE_DIR
                )
                self.model = CLIPModel.from_pretrained(
                    self.model_path,
                    cache_dir=CLIP_CACHE_DIR
                )
                logger.info(f"Loaded pre-trained CLIP model: {self.model_path}")
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"CLIP model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
    
    def compute_similarity(
        self,
        image_path: str,
        text: str
    ) -> float:
        """
        Compute similarity score between image and text.
        
        Args:
            image_path: Path to image file
            text: Text description to compare
            
        Returns:
            Similarity score (0-1), where higher values indicate better match
            
        Raises:
            RuntimeError: If inference fails
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess inputs
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Compute features
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get similarity score using cosine similarity
                # CLIP outputs logits_per_image which are already similarity scores
                # scaled by a learned temperature parameter
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity (range: -1 to 1)
                cosine_similarity = (image_embeds * text_embeds).sum(dim=-1)
                
                # Normalize to 0-1 range for easier interpretation
                # (cosine_similarity + 1) / 2 maps [-1, 1] to [0, 1]
                similarity_score = ((cosine_similarity + 1) / 2).item()
            
            logger.info(f"CLIP similarity score: {similarity_score:.4f}")
            return float(similarity_score)
            
        except Exception as e:
            logger.error(f"Error computing CLIP similarity: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to compute similarity: {str(e)}")
    
    def detect_mismatch(
        self,
        image_path: str,
        text: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Detect if there's a mismatch between image and text.
        
        Args:
            image_path: Path to image file
            text: Text description to compare
            threshold: Custom threshold (uses config default if None)
            
        Returns:
            Tuple of (is_match, similarity_score, decision_text)
            - is_match: True if image matches text, False if mismatch
            - similarity_score: Computed similarity score (0-1)
            - decision_text: Human-readable explanation
        """
        # Use provided threshold or default
        threshold = threshold if threshold is not None else CLIP_SIMILARITY_THRESHOLD
        
        # Compute similarity
        similarity_score = self.compute_similarity(image_path, text)
        
        # Determine match/mismatch
        is_match = similarity_score >= threshold
        
        # Generate decision text
        if is_match:
            decision_text = f"Match (score: {similarity_score:.3f} >= {threshold:.3f})"
        else:
            decision_text = f"Mismatch (score: {similarity_score:.3f} < {threshold:.3f})"
        
        logger.info(f"CLIP mismatch detection: {decision_text}")
        
        return is_match, similarity_score, decision_text


# Global instance (lazy loaded)
_clip_detector_instance = None


def get_clip_detector(model_path: Optional[str] = None) -> CLIPMismatchDetector:
    """
    Get or create global CLIP detector instance.
    
    Args:
        model_path: Optional path to fine-tuned model
        
    Returns:
        CLIPMismatchDetector instance
    """
    global _clip_detector_instance
    
    # Create instance if it doesn't exist or if a custom model path is requested
    if _clip_detector_instance is None or model_path is not None:
        _clip_detector_instance = CLIPMismatchDetector(model_path=model_path)
    
    return _clip_detector_instance


def check_mismatch_clip(
    image_path: str,
    description: str,
    threshold: Optional[float] = None
) -> dict:
    """
    Convenience function to check for image-description mismatch using CLIP.
    
    Args:
        image_path: Path to image file
        description: Text description to compare
        threshold: Optional custom threshold
        
    Returns:
        Dictionary with mismatch detection results:
        {
            'is_match': bool,
            'similarity_score': float,
            'decision': str,
            'threshold_used': float
        }
    """
    if not description or description.strip() == "":
        return {
            'is_match': None,
            'similarity_score': None,
            'decision': 'No description provided',
            'threshold_used': threshold or CLIP_SIMILARITY_THRESHOLD
        }
    
    try:
        detector = get_clip_detector()
        threshold = threshold if threshold is not None else CLIP_SIMILARITY_THRESHOLD
        is_match, similarity_score, decision = detector.detect_mismatch(
            image_path,
            description,
            threshold
        )
        
        return {
            'is_match': is_match,
            'similarity_score': similarity_score,
            'decision': decision,
            'threshold_used': threshold
        }
    except Exception as e:
        logger.error(f"Error in CLIP mismatch check: {str(e)}", exc_info=True)
        return {
            'is_match': None,
            'similarity_score': None,
            'decision': f'Error: {str(e)}',
            'threshold_used': threshold or CLIP_SIMILARITY_THRESHOLD
        }
