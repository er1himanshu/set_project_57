"""
CLIP-based image-text similarity service for detecting mismatches 
between product images and their descriptions.
"""

import logging
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Optional, Tuple
from ..config import CLIP_MODEL_NAME, CLIP_SIMILARITY_THRESHOLD, CLIP_DEVICE, CLIP_ZERO_SHOT_LABELS

logger = logging.getLogger(__name__)


class CLIPService:
    """
    Service for computing image-text similarity using CLIP models.
    Supports match/mismatch detection and zero-shot classification.
    """
    
    _instance = None
    _model = None
    _processor = None
    _device = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super(CLIPService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize CLIP model and processor (lazy loading)."""
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load CLIP model and processor."""
        try:
            logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
            self._device = CLIP_DEVICE if torch.cuda.is_available() or CLIP_DEVICE == "cpu" else "cpu"
            
            # Load model and processor
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            
            # Move model to device
            self._model.to(self._device)
            self._model.eval()  # Set to evaluation mode
            
            logger.info(f"CLIP model loaded successfully on device: {self._device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}", exc_info=True)
            raise
    
    def compute_similarity(self, image_path: str, text: str) -> float:
        """
        Compute cosine similarity between image and text.
        
        Args:
            image_path: Path to the image file
            text: Text description to compare
            
        Returns:
            Similarity score between 0 and 1 (higher means more similar)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process inputs
            inputs = self._processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Compute features
            with torch.no_grad():
                outputs = self._model(**inputs)
                
                # Get image and text embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_embeds @ text_embeds.T).squeeze().item()
                
                # Scale to 0-1 range (cosine similarity is -1 to 1)
                similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}", exc_info=True)
            return 0.0
    
    def detect_mismatch(
        self, 
        image_path: str, 
        text: str, 
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Detect if there's a mismatch between image and text description.
        
        Args:
            image_path: Path to the image file
            text: Text description to compare
            threshold: Custom threshold (uses config default if None)
            
        Returns:
            Tuple of (is_match, similarity_score, status_message)
            - is_match: True if similarity >= threshold
            - similarity_score: The computed similarity (0-1)
            - status_message: Human-readable status
        """
        if not text or text.strip() == "":
            return (True, 0.0, "No description provided")
        
        threshold = threshold if threshold is not None else CLIP_SIMILARITY_THRESHOLD
        similarity = self.compute_similarity(image_path, text)
        
        is_match = similarity >= threshold
        
        if is_match:
            status = f"Match (score: {similarity:.3f})"
        else:
            status = f"Mismatch detected (score: {similarity:.3f}, threshold: {threshold})"
        
        return (is_match, similarity, status)
    
    def zero_shot_classify(
        self, 
        image_path: str, 
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Perform zero-shot classification on the image using provided labels.
        
        Args:
            image_path: Path to the image file
            labels: List of label texts (uses config default if None)
            
        Returns:
            Dictionary mapping labels to their probability scores
        """
        labels = labels if labels is not None else CLIP_ZERO_SHOT_LABELS
        
        if not labels:
            logger.warning("No labels provided for zero-shot classification")
            return {}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process inputs with multiple text candidates
            inputs = self._processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Compute features
            with torch.no_grad():
                outputs = self._model(**inputs)
                
                # Get probabilities for each label
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Create label -> probability mapping
            results = {
                label: float(prob.item()) 
                for label, prob in zip(labels, probs)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in zero-shot classification: {str(e)}", exc_info=True)
            return {}
    
    def batch_compute_similarity(
        self, 
        image_paths: List[str], 
        texts: List[str]
    ) -> List[float]:
        """
        Compute similarities for multiple image-text pairs efficiently.
        
        Args:
            image_paths: List of image file paths
            texts: List of text descriptions (same length as image_paths)
            
        Returns:
            List of similarity scores
        """
        if len(image_paths) != len(texts):
            raise ValueError("image_paths and texts must have the same length")
        
        similarities = []
        for image_path, text in zip(image_paths, texts):
            similarity = self.compute_similarity(image_path, text)
            similarities.append(similarity)
        
        return similarities


# Global instance for reuse
_clip_service = None


def get_clip_service() -> CLIPService:
    """
    Get or create the global CLIP service instance.
    
    Returns:
        CLIPService instance
    """
    global _clip_service
    if _clip_service is None:
        _clip_service = CLIPService()
    return _clip_service


def analyze_image_text_match(
    image_path: str, 
    description: str,
    threshold: Optional[float] = None,
    use_zero_shot: bool = False,
    zero_shot_labels: Optional[List[str]] = None
) -> Dict:
    """
    Main function to analyze image-text matching using CLIP.
    
    Args:
        image_path: Path to the image file
        description: Text description to compare
        threshold: Custom similarity threshold
        use_zero_shot: Whether to perform zero-shot classification
        zero_shot_labels: Custom labels for zero-shot classification
        
    Returns:
        Dictionary with analysis results including:
        - is_match: Boolean indicating match/mismatch
        - similarity_score: Float similarity score (0-1)
        - status: Human-readable status message
        - zero_shot_results: Optional dict of zero-shot classification results
    """
    try:
        service = get_clip_service()
        
        # Compute match/mismatch
        is_match, similarity, status = service.detect_mismatch(
            image_path, description, threshold
        )
        
        results = {
            "is_match": is_match,
            "similarity_score": similarity,
            "status": status
        }
        
        # Optional zero-shot classification
        if use_zero_shot:
            zero_shot_results = service.zero_shot_classify(image_path, zero_shot_labels)
            results["zero_shot_results"] = zero_shot_results
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analyze_image_text_match: {str(e)}", exc_info=True)
        return {
            "is_match": True,
            "similarity_score": 0.0,
            "status": f"Error: {str(e)}",
            "zero_shot_results": {}
        }
