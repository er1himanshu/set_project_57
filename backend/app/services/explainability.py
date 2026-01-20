"""
CLIP Explainability using Vision Transformer (ViT) Attention Rollout

This module provides functionality to visualize which parts of an image
contribute most to CLIP's similarity score using attention rollout from
the Vision Transformer layers.

Graceful Fallback:
When attention tensors are not available (e.g., on some CLIP model setups),
the module automatically falls back to a similarity-only explanation without
attention visualization. This ensures the explainability feature always returns
a meaningful response rather than failing with an error.
"""

import logging
import base64
import io
from typing import Tuple, Optional
import numpy as np
import torch
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel

from .mismatch_detector import get_clip_model, MismatchDetectionUnavailableError
from ..config import CLIP_MODEL_NAME

logger = logging.getLogger(__name__)


def compute_attention_rollout(attentions: torch.Tensor, discard_ratio: float = 0.9) -> np.ndarray:
    """
    Compute attention rollout from ViT attention maps.
    
    Attention rollout progressively multiplies attention weights across layers,
    identifying which image patches most influence the final representation.
    
    Args:
        attentions: Attention weights from all transformer layers
                   Shape: (num_layers, num_heads, num_patches, num_patches)
        discard_ratio: Ratio of lowest attention weights to discard per layer (0-1)
    
    Returns:
        np.ndarray: Rolled out attention map (num_patches,)
        
    Raises:
        ValueError: If attentions tensor is None or has invalid dimensions
    """
    # Guard: Check for None or empty tensor
    if attentions is None:
        raise ValueError("Attention tensor cannot be None")
    
    if attentions.numel() == 0:
        raise ValueError("Attention tensor is empty (expected tensor with elements > 0)")
    
    # Average attention across all heads
    attentions = torch.mean(attentions, dim=1)  # (num_layers, num_patches, num_patches)
    
    # Add residual connections (identity matrix)
    residual_att = torch.eye(attentions.size(1)).to(attentions.device)
    aug_att_mat = attentions + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)
    
    # Progressively multiply attention matrices through layers (rollout)
    joint_attentions = torch.zeros(aug_att_mat.size()).to(attentions.device)
    joint_attentions[0] = aug_att_mat[0]
    
    for i in range(1, aug_att_mat.size(0)):
        joint_attentions[i] = torch.matmul(aug_att_mat[i], joint_attentions[i-1])
    
    # Get attention from CLS token to all patches (last layer)
    # CLS token is at index 0
    v = joint_attentions[-1]
    attention_map = v[0, 1:]  # Exclude CLS token itself
    
    return attention_map.cpu().numpy()


def create_heatmap_overlay(
    image_path: str,
    attention_map: np.ndarray,
    grid_size: int = 7,
    alpha: float = 0.5
) -> Image.Image:
    """
    Create a heatmap overlay on the original image based on attention weights.
    
    Args:
        image_path: Path to the original image
        attention_map: 1D array of attention weights for each patch
        grid_size: Size of the attention grid (e.g., 7x7 for 49 patches)
        alpha: Transparency of heatmap overlay (0=transparent, 1=opaque)
    
    Returns:
        PIL.Image: Image with heatmap overlay
    """
    # Load original image
    original_img = Image.open(image_path).convert("RGB")
    img_array = np.array(original_img)
    
    # Reshape attention map to grid
    attention_grid = attention_map.reshape(grid_size, grid_size)
    
    # Normalize attention values to 0-255 range
    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
    attention_grid = (attention_grid * 255).astype(np.uint8)
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(attention_grid, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Apply colormap (COLORMAP_JET for red-yellow-blue gradient)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlayed)


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        str: Base64-encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def generate_fallback_explanation(
    image_path: str,
    description: str,
    similarity_score: float,
    threshold: Optional[float] = None
) -> dict:
    """
    Generate a fallback explanation without attention rollout.
    
    This is used when attention tensors are not available from the CLIP model.
    Instead of a heatmap, we provide a simplified explanation with the similarity score.
    
    Args:
        image_path: Path to the image file
        description: Text description
        similarity_score: CLIP similarity score
        threshold: Optional similarity threshold
    
    Returns:
        dict: Fallback explanation with similarity score (no heatmap)
    """
    # Load and encode the original image (no heatmap overlay)
    image = Image.open(image_path).convert("RGB")
    # Resize image for consistent output size
    max_size = 500
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    image_base64 = encode_image_to_base64(image, format="PNG")
    
    # Determine if mismatch
    threshold = threshold if threshold else 0.25
    is_mismatch = similarity_score < threshold
    
    # Generate message
    if is_mismatch:
        message = f"Mismatch detected (score: {similarity_score:.2f})"
    else:
        message = f"Match confirmed (score: {similarity_score:.2f})"
    
    # Generate fallback explanation
    explanation = (
        "CLIP similarity analysis completed successfully. "
        "Note: Detailed attention visualization is not available on this system. "
        "The similarity score indicates how well the image matches the description, "
        "with higher scores indicating better alignment between image and text."
    )
    
    logger.info(f"Generated fallback CLIP explanation (no attention): {message}")
    
    return {
        "similarity_score": float(similarity_score),
        "has_mismatch": is_mismatch,
        "message": message,
        "heatmap_base64": image_base64,
        "explanation": explanation,
        "attention_available": False
    }


def generate_clip_explanation(
    image_path: str,
    description: str,
    threshold: Optional[float] = None
) -> dict:
    """
    Generate CLIP explanation with attention rollout heatmap.
    
    Falls back to similarity-only explanation if attention is unavailable.
    
    Args:
        image_path: Path to the image file
        description: Text description to compare with image
        threshold: Optional similarity threshold
    
    Returns:
        dict: Explanation results including similarity score and heatmap (or fallback)
    
    Raises:
        MismatchDetectionUnavailableError: If CLIP model is not available
    """
    # Validate inputs
    if not description or description.strip() == "":
        raise ValueError("Description is required for explanation generation")
    
    try:
        # Load CLIP model and processor
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
        
        # Get model outputs with attention weights
        # Note: output_attentions is supported by CLIP ViT models
        attention_supported = True
        with torch.no_grad():
            try:
                outputs = model(**inputs, output_attentions=True)
            except TypeError:
                # Model may not support output_attentions parameter
                logger.info("CLIP model does not support attention output parameter, using fallback")
                attention_supported = False
                outputs = model(**inputs)
        
        # Calculate similarity score
        logits_per_image = outputs.logits_per_image
        similarity_score = torch.sigmoid(logits_per_image / 100.0).item()
        
        # Try to extract vision attention weights if supported
        vision_attentions = None
        if attention_supported:
            try:
                vision_attentions = outputs.vision_model_output.attentions
            except AttributeError:
                logger.info("CLIP model output does not contain attention data, using fallback")
                attention_supported = False
        
        # Check if attention outputs are available and valid
        attention_available = (
            attention_supported and 
            vision_attentions is not None and 
            len(vision_attentions) > 0 and
            not any(att is None for att in vision_attentions)
        )
        
        # Use fallback if attention is not available
        if not attention_available:
            logger.info("Attention visualization unavailable, generating fallback explanation")
            return generate_fallback_explanation(
                image_path=image_path,
                description=description,
                similarity_score=similarity_score,
                threshold=threshold
            )
        
        # Stack attention tensors from all layers
        # Shape: (num_layers, batch_size, num_heads, num_patches, num_patches)
        attention_stack = torch.stack(vision_attentions)
        attention_stack = attention_stack.squeeze(1)  # Remove batch dimension
        
        # Compute attention rollout
        attention_map = compute_attention_rollout(attention_stack)
        
        # Determine grid size from model architecture
        # For CLIP ViT models, patches are arranged in a square grid
        # Total patches = (image_size / patch_size) ^ 2
        # For ViT-B/32: 224/32 = 7, so 7x7 grid
        import math
        num_patches = attention_map.shape[0]
        grid_size = int(math.sqrt(num_patches))
        
        # Create heatmap overlay
        heatmap_image = create_heatmap_overlay(
            image_path=image_path,
            attention_map=attention_map,
            grid_size=grid_size,
            alpha=0.5
        )
        
        # Encode heatmap to base64
        heatmap_base64 = encode_image_to_base64(heatmap_image, format="PNG")
        
        # Determine if mismatch
        is_mismatch = similarity_score < (threshold if threshold else 0.25)
        
        # Generate message
        if is_mismatch:
            message = f"Mismatch detected (score: {similarity_score:.2f})"
        else:
            message = f"Match confirmed (score: {similarity_score:.2f})"
        
        logger.info(f"Generated CLIP explanation: {message}")
        
        return {
            "similarity_score": float(similarity_score),
            "has_mismatch": is_mismatch,
            "message": message,
            "heatmap_base64": heatmap_base64,
            "explanation": "Heatmap shows which image regions most influenced the similarity score. Warmer colors (red/yellow) indicate higher attention.",
            "attention_available": True
        }
        
    except MismatchDetectionUnavailableError:
        logger.warning("CLIP model unavailable for explanation generation")
        raise
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}") from e
    except Exception as e:
        logger.error(f"Error generating CLIP explanation: {type(e).__name__}: {str(e)}")
        raise Exception(f"Failed to generate explanation: {str(e)}") from e
