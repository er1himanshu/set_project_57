# CLIP Integration Examples

This directory contains example scripts demonstrating the CLIP-based image-text similarity features.

## Available Examples

### clip_examples.py

Comprehensive demonstration of CLIP functionality including:

1. **Basic Similarity**: Computing similarity scores between images and text
2. **Mismatch Detection**: Automatic detection of image-text mismatches
3. **Full Analysis**: Complete analysis workflow
4. **Zero-Shot Classification**: Classifying images into categories without training
5. **Batch Processing**: Efficiently processing multiple image-text pairs
6. **Custom Thresholds**: Testing different similarity thresholds

## Running the Examples

### Prerequisites

Make sure dependencies are installed:
```bash
cd backend
pip install -r requirements.txt
```

### Run All Examples

```bash
cd backend
python examples/clip_examples.py
```

**Note**: The first run will download the CLIP model (~500MB) which may take a few minutes depending on your internet connection. Subsequent runs will be much faster as the model is cached.

## Example Output

```
============================================================
CLIP Integration - Sample Usage Examples
============================================================

📝 Note: These examples use simple colored squares.
   Real product images will show more nuanced results.
   The first run may take longer as the model downloads.

============================================================
Example 1: Basic Image-Text Similarity
============================================================
✓ Red image vs 'a red colored square': 0.2847
✗ Red image vs 'a blue colored square': 0.2435

Expected: Matching description has higher score
Result: ✅ PASS

============================================================
Example 2: Match/Mismatch Detection
============================================================

✓ Green image + 'green colored square':
  Is Match: True
  Score: 0.2789
  Status: Match (score: 0.279)

✗ Green image + 'red colored square':
  Is Match: False
  Score: 0.2412
  Status: Mismatch detected (score: 0.241, threshold: 0.25)

...
```

## Customizing Examples

You can modify the examples to test with your own images:

```python
# Instead of creating sample images
image_path = "path/to/your/product.jpg"

# Test with your description
description = "Red cotton t-shirt with round neck"

# Run analysis
result = analyze_image_text_match(image_path, description)
print(f"Score: {result['similarity_score']:.4f}")
print(f"Match: {result['is_match']}")
```

## Testing with Real Product Images

To test with actual product images:

1. Place your product images in a directory
2. Create a test script:

```python
from app.services.clip_service import get_clip_service

service = get_clip_service()

# Test your product
similarity = service.compute_similarity(
    "products/red_shirt.jpg",
    "Red cotton t-shirt"
)

print(f"Similarity: {similarity:.4f}")
```

## Next Steps

- Review the [CLIP Integration Guide](../../docs/CLIP_INTEGRATION.md)
- Try the CLI tool: `python clip_cli.py --help`
- Fine-tune on your data: See [Dataset Format](../training/datasets/README.md)

## Troubleshooting

### Model Download Issues

If the model fails to download:
```bash
# Manually download the model
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Import Errors

Make sure you're running from the backend directory:
```bash
cd backend
python examples/clip_examples.py
```

### Memory Issues

If you encounter memory issues:
- Close other applications
- Use a smaller CLIP model in `config.py`
- Reduce batch sizes in batch operations
