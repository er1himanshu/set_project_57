"""
CLIP Explainability using Attention Rollout

This module provides functionality to generate attention-based heatmaps from CLIP's
Vision Transformer (ViT) to explain which regions of an image contribute most to
the image-text similarity score.
"""

import logging
import io
import base64
import numpy as np
import torch
from PIL import Image
import cv2
from typing import Tuple, Optional

from .mismatch_detector import get_clip_model, MismatchDetectionUnavailableError

logger = logging.getLogger(__name__)


def get_attention_rollout(attention_maps: torch.Tensor, discard_ratio: float = 0.1) -> np.ndarray:
    """
    Compute attention rollout from attention maps of all layers.
    
    Attention rollout progressively multiplies attention maps across layers to track
    how information flows from input patches to output, revealing which image regions
    contribute most to the final representation.
    
    Args:
        attention_maps: Attention weights from all transformer layers
                       Shape: (num_layers, num_heads, num_patches, num_patches)
        discard_ratio: Ratio of lowest attention values to discard (default 0.1)
        
    Returns:
        np.ndarray: Attention rollout map for the [CLS] token, shape (num_patches,)
    """
    # Average attention across all heads for each layer
    # Shape: (num_layers, num_patches, num_patches)
    attention_maps = attention_maps.mean(dim=1)
    
    # Add identity matrix to account for residual connections
    # This ensures each token attends to itself
    num_patches = attention_maps.shape[-1]
    eye_matrix = torch.eye(num_patches).to(attention_maps.device)
    attention_maps = attention_maps + eye_matrix
    
    # Normalize rows to sum to 1 (make them probability distributions)
    attention_maps = attention_maps / attention_maps.sum(dim=-1, keepdim=True)
    
    # Apply attention rollout: multiply attention matrices across layers
    rollout = attention_maps[0]
    for layer_attention in attention_maps[1:]:
        rollout = torch.matmul(rollout, layer_attention)
    
    # Extract attention from [CLS] token (first token) to all patches
    # This shows which patches influenced the final representation
    cls_attention = rollout[0, 1:]  # Skip [CLS] token itself, take patch tokens
    
    return cls_attention.cpu().numpy()


def generate_attention_heatmap(
    image_path: str,
    description: str,
    discard_ratio: float = 0.1
) -> Tuple[Optional[str], Optional[float], str]:
    """
    Generate attention-based heatmap showing which image regions contribute to
    the CLIP similarity score with the given description.
    
    Args:
        image_path: Path to the image file
        description: Text description to compare against
        discard_ratio: Ratio of lowest attention values to discard
        
    Returns:
        Tuple containing:
            - heatmap_base64: Base64-encoded PNG image of heatmap overlay, or None
            - similarity_score: CLIP similarity score (0-1), or None
            - message: Status message
    """
    # Validate description
    if not description or len(description.strip()) < 10:
        return None, None, "Description must be at least 10 characters"
    
    try:
        # Load CLIP model and processor
        model, processor = get_clip_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        # Process inputs
        inputs = processor(
            text=[description],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Calculate similarity score
        logits_per_image = outputs.logits_per_image
        similarity_score = torch.sigmoid(logits_per_image / 100.0).item()
        
        # Extract vision attention maps
        # outputs.vision_model_output.attentions is a tuple of attention tensors
        # Each tensor has shape: (batch, num_heads, num_patches, num_patches)
        vision_attentions = outputs.vision_model_output.attentions
        
        if vision_attentions is None or len(vision_attentions) == 0:
            logger.warning("No attention maps available from CLIP model")
            return None, similarity_score, f"Attention maps unavailable (score: {similarity_score:.2f})"
        
        # Stack attention maps from all layers
        # Shape: (num_layers, batch, num_heads, num_patches, num_patches)
        attention_stack = torch.stack(vision_attentions)
        
        # Remove batch dimension (we only process one image)
        # Shape: (num_layers, num_heads, num_patches, num_patches)
        attention_stack = attention_stack[:, 0, :, :, :]
        
        # Compute attention rollout
        attention_weights = get_attention_rollout(attention_stack, discard_ratio)
        
        # Reshape attention weights to 2D grid
        # CLIP ViT-B/32 uses 7x7 patches (224/32 = 7)
        grid_size = int(np.sqrt(len(attention_weights)))
        attention_map = attention_weights.reshape(grid_size, grid_size)
        
        # Normalize to 0-1 range
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize heatmap to original image size using bilinear interpolation
        attention_map_resized = cv2.resize(
            attention_map,
            original_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert to heatmap (colormap)
        heatmap = cv2.applyColorMap(
            (attention_map_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay: blend heatmap with original image
        image_np = np.array(image)
        overlay = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)
        
        # Convert to PIL Image and encode as base64 PNG
        overlay_pil = Image.fromarray(overlay.astype(np.uint8))
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format='PNG')
        buffer.seek(0)
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        message = f"Explanation generated (similarity: {similarity_score:.2f})"
        logger.info(f"Generated attention heatmap: {message}")
        
        return heatmap_base64, float(similarity_score), message
        
    except MismatchDetectionUnavailableError:
        logger.warning("CLIP model unavailable for explainability")
        return None, None, "CLIP model is not available"
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error generating attention heatmap: {type(e).__name__}: {str(e)}")
        return None, None, f"Failed to generate explanation: {type(e).__name__}"
