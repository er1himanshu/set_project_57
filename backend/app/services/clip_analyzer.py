"""
CLIP-based image-text similarity analysis for mismatch detection.

This module provides utilities for computing image-text similarity using CLIP models,
detecting mismatches between product images and descriptions, and performing 
class-based comparisons against category labels.
"""

import torch
import logging
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from transformers import CLIPProcessor, CLIPModel
from ..config import (
    CLIP_MODEL_NAME, 
    CLIP_MODEL_PATH, 
    CLIP_SIMILARITY_THRESHOLD,
    CLIP_BATCH_SIZE,
    DEFAULT_CATEGORIES
)

logger = logging.getLogger(__name__)


class CLIPAnalyzer:
    """
    CLIP-based analyzer for image-text similarity and mismatch detection.
    
    This class provides zero-shot image-text matching capabilities using 
    pre-trained or fine-tuned CLIP models.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize CLIP analyzer with model and processor.
        
        Args:
            model_path: Optional path to fine-tuned model. If None, uses default CLIP model.
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing CLIP analyzer on device: {self.device}")
        
        # Load model path from config if not provided
        if model_path is None:
            model_path = CLIP_MODEL_PATH
        
        try:
            if model_path:
                # Load fine-tuned model
                logger.info(f"Loading fine-tuned CLIP model from: {model_path}")
                self.model = CLIPModel.from_pretrained(model_path).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_path)
            else:
                # Load default pre-trained model
                logger.info(f"Loading pre-trained CLIP model: {CLIP_MODEL_NAME}")
                self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            
            self.model.eval()
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}", exc_info=True)
            raise
    
    def compute_similarity(
        self, 
        image_path: str, 
        text: str
    ) -> float:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_path: Path to image file
            text: Text description to compare
            
        Returns:
            float: Similarity score between 0 and 1 (higher = more similar)
            
        Raises:
            Exception: If image cannot be loaded or processing fails
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Compute embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize embeddings
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_embeds @ text_embeds.T).item()
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}", exc_info=True)
            raise
    
    def detect_mismatch(
        self,
        image_path: str,
        text: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Detect if there's a mismatch between image and text description.
        
        Args:
            image_path: Path to image file
            text: Text description to check
            threshold: Similarity threshold (default from config). 
                      Scores below threshold indicate mismatch.
            
        Returns:
            dict: {
                'is_mismatch': bool,
                'similarity_score': float,
                'threshold_used': float,
                'confidence': str ('high', 'medium', 'low')
            }
        """
        if threshold is None:
            threshold = CLIP_SIMILARITY_THRESHOLD
        
        try:
            similarity = self.compute_similarity(image_path, text)
            is_mismatch = similarity < threshold
            
            # Determine confidence level
            distance_from_threshold = abs(similarity - threshold)
            if distance_from_threshold > 0.15:
                confidence = 'high'
            elif distance_from_threshold > 0.05:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'is_mismatch': is_mismatch,
                'similarity_score': similarity,
                'threshold_used': threshold,
                'confidence': confidence,
                'match_quality': self._get_match_quality(similarity)
            }
            
        except Exception as e:
            logger.error(f"Error detecting mismatch: {str(e)}", exc_info=True)
            raise
    
    def class_based_comparison(
        self,
        image_path: str,
        categories: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Compare image against a list of category labels (zero-shot classification).
        
        Args:
            image_path: Path to image file
            categories: List of category labels to compare against. 
                       If None, uses DEFAULT_CATEGORIES from config.
            top_k: Number of top matching categories to return
            
        Returns:
            list: List of dicts with 'category' and 'score' keys, 
                 sorted by score (descending)
        """
        if categories is None:
            categories = DEFAULT_CATEGORIES
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare text prompts with template for better zero-shot performance
            texts = [f"a photo of a {category}" for category in categories]
            
            # Process inputs
            inputs = self.processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Compute similarities
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Get top-k results
            top_k = min(top_k, len(categories))
            top_probs, top_indices = torch.topk(probs, top_k)
            
            results = [
                {
                    'category': categories[idx.item()],
                    'score': prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in class-based comparison: {str(e)}", exc_info=True)
            raise
    
    def batch_analyze(
        self,
        image_paths: List[str],
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Union[bool, float, str]]]:
        """
        Batch process multiple image-text pairs for efficiency.
        
        Args:
            image_paths: List of image file paths
            texts: List of text descriptions (must match length of image_paths)
            batch_size: Batch size for processing (default from config)
            
        Returns:
            list: List of mismatch detection results for each pair
            
        Raises:
            ValueError: If lengths of image_paths and texts don't match
        """
        if len(image_paths) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(image_paths)} images vs {len(texts)} texts"
            )
        
        if batch_size is None:
            batch_size = CLIP_BATCH_SIZE
        
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_images = image_paths[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            
            for img_path, text in zip(batch_images, batch_texts):
                try:
                    result = self.detect_mismatch(img_path, text)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
                    results.append({
                        'is_mismatch': None,
                        'similarity_score': None,
                        'threshold_used': CLIP_SIMILARITY_THRESHOLD,
                        'confidence': 'error',
                        'error': str(e)
                    })
        
        return results
    
    @staticmethod
    def _get_match_quality(similarity: float) -> str:
        """
        Get qualitative description of match quality based on similarity score.
        
        Args:
            similarity: Similarity score between 0 and 1
            
        Returns:
            str: Quality description ('excellent', 'good', 'fair', 'poor', 'mismatch')
        """
        if similarity >= 0.4:
            return 'excellent'
        elif similarity >= 0.3:
            return 'good'
        elif similarity >= 0.25:
            return 'fair'
        elif similarity >= 0.15:
            return 'poor'
        else:
            return 'mismatch'


# Global analyzer instance (lazy initialization)
_analyzer_instance: Optional[CLIPAnalyzer] = None


def get_clip_analyzer(
    model_path: Optional[str] = None,
    force_reload: bool = False
) -> CLIPAnalyzer:
    """
    Get or create global CLIP analyzer instance (singleton pattern).
    
    Args:
        model_path: Optional path to fine-tuned model
        force_reload: If True, reload the model even if already initialized
        
    Returns:
        CLIPAnalyzer: Global analyzer instance
    """
    global _analyzer_instance
    
    if _analyzer_instance is None or force_reload:
        _analyzer_instance = CLIPAnalyzer(model_path=model_path)
    
    return _analyzer_instance
