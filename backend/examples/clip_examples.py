#!/usr/bin/env python3
"""
Sample usage examples for CLIP-based image-text similarity analysis.

This script demonstrates various use cases of the CLIP integration,
including basic similarity checking, mismatch detection, and zero-shot
classification.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clip_service import (
    get_clip_service,
    analyze_image_text_match
)
from PIL import Image
import numpy as np


def create_sample_image(color=(255, 0, 0), size=(224, 224), filename="sample.jpg"):
    """Create a simple colored sample image for testing."""
    img = Image.new('RGB', size, color)
    img.save(filename)
    return filename


def example_1_basic_similarity():
    """Example 1: Basic image-text similarity computation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Image-Text Similarity")
    print("=" * 60)
    
    # Create a red image
    image_path = create_sample_image(color=(255, 0, 0), filename="/tmp/red_sample.jpg")
    
    service = get_clip_service()
    
    # Test with matching description
    similarity_match = service.compute_similarity(image_path, "a red colored square")
    print(f"✓ Red image vs 'a red colored square': {similarity_match:.4f}")
    
    # Test with non-matching description
    similarity_mismatch = service.compute_similarity(image_path, "a blue colored square")
    print(f"✗ Red image vs 'a blue colored square': {similarity_mismatch:.4f}")
    
    print(f"\nExpected: Matching description has higher score")
    print(f"Result: {'✅ PASS' if similarity_match > similarity_mismatch else '❌ FAIL'}")
    
    # Cleanup
    os.remove(image_path)


def example_2_mismatch_detection():
    """Example 2: Automatic match/mismatch detection."""
    print("\n" + "=" * 60)
    print("Example 2: Match/Mismatch Detection")
    print("=" * 60)
    
    # Create a green image
    image_path = create_sample_image(color=(0, 255, 0), filename="/tmp/green_sample.jpg")
    
    # Test matching description
    is_match, score, status = get_clip_service().detect_mismatch(
        image_path, 
        "a green colored square"
    )
    print(f"\n✓ Green image + 'green colored square':")
    print(f"  Is Match: {is_match}")
    print(f"  Score: {score:.4f}")
    print(f"  Status: {status}")
    
    # Test mismatching description
    is_match, score, status = get_clip_service().detect_mismatch(
        image_path,
        "a red colored square"
    )
    print(f"\n✗ Green image + 'red colored square':")
    print(f"  Is Match: {is_match}")
    print(f"  Score: {score:.4f}")
    print(f"  Status: {status}")
    
    # Cleanup
    os.remove(image_path)


def example_3_full_analysis():
    """Example 3: Full analysis with all features."""
    print("\n" + "=" * 60)
    print("Example 3: Full Analysis")
    print("=" * 60)
    
    # Create a blue image
    image_path = create_sample_image(color=(0, 0, 255), filename="/tmp/blue_sample.jpg")
    
    # Full analysis
    result = analyze_image_text_match(
        image_path=image_path,
        description="a blue colored square",
        threshold=0.25,
        use_zero_shot=False
    )
    
    print(f"\nAnalysis Results:")
    print(f"  Is Match: {result['is_match']}")
    print(f"  Similarity Score: {result['similarity_score']:.4f}")
    print(f"  Status: {result['status']}")
    
    # Cleanup
    os.remove(image_path)


def example_4_zero_shot_classification():
    """Example 4: Zero-shot classification with custom labels."""
    print("\n" + "=" * 60)
    print("Example 4: Zero-Shot Classification")
    print("=" * 60)
    
    # Create a yellow image
    image_path = create_sample_image(color=(255, 255, 0), filename="/tmp/yellow_sample.jpg")
    
    # Define custom labels
    labels = [
        "a red object",
        "a green object",
        "a blue object",
        "a yellow object",
        "a white object"
    ]
    
    service = get_clip_service()
    results = service.zero_shot_classify(image_path, labels)
    
    print(f"\nClassification Results:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for label, prob in sorted_results:
        bar = "█" * int(prob * 50)
        print(f"  {label:20s}: {prob:.4f} {bar}")
    
    print(f"\nTop prediction: {sorted_results[0][0]}")
    
    # Cleanup
    os.remove(image_path)


def example_5_batch_processing():
    """Example 5: Batch processing multiple images."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    # Create multiple sample images
    images = []
    texts = []
    colors = [
        ((255, 0, 0), "red"),
        ((0, 255, 0), "green"),
        ((0, 0, 255), "blue")
    ]
    
    for i, (color, name) in enumerate(colors):
        img_path = create_sample_image(color=color, filename=f"/tmp/{name}_sample.jpg")
        images.append(img_path)
        texts.append(f"a {name} colored square")
    
    # Batch compute similarities
    service = get_clip_service()
    similarities = service.batch_compute_similarity(images, texts)
    
    print(f"\nBatch Similarity Results:")
    for i, (img, text, sim) in enumerate(zip(images, texts, similarities)):
        color_name = colors[i][1]
        print(f"  {color_name.title()} image + '{text}': {sim:.4f}")
    
    # Cleanup
    for img in images:
        os.remove(img)


def example_6_custom_threshold():
    """Example 6: Using custom similarity thresholds."""
    print("\n" + "=" * 60)
    print("Example 6: Custom Threshold Testing")
    print("=" * 60)
    
    # Create a magenta image
    image_path = create_sample_image(color=(255, 0, 255), filename="/tmp/magenta_sample.jpg")
    
    description = "a purple colored square"
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    
    print(f"\nTesting different thresholds:")
    print(f"Image: Magenta")
    print(f"Description: '{description}'")
    print(f"\n{'Threshold':<12} {'Match':<8} {'Score':<10}")
    print("-" * 30)
    
    for threshold in thresholds:
        is_match, score, status = get_clip_service().detect_mismatch(
            image_path,
            description,
            threshold=threshold
        )
        match_str = "✓ Match" if is_match else "✗ Miss"
        print(f"{threshold:<12.2f} {match_str:<8} {score:.4f}")
    
    # Cleanup
    os.remove(image_path)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CLIP Integration - Sample Usage Examples")
    print("=" * 60)
    
    try:
        print("\n📝 Note: These examples use simple colored squares.")
        print("   Real product images will show more nuanced results.")
        print("   The first run may take longer as the model downloads.")
        
        example_1_basic_similarity()
        example_2_mismatch_detection()
        example_3_full_analysis()
        example_4_zero_shot_classification()
        example_5_batch_processing()
        example_6_custom_threshold()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nFor more information, see:")
        print("  - docs/CLIP_INTEGRATION.md")
        print("  - backend/training/datasets/README.md")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
