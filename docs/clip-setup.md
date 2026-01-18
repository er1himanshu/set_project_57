# CLIP Setup Guide

This guide covers the installation and configuration of the CLIP-based image-text mismatch detection feature.

## Overview

The CLIP integration provides:
- **Zero-shot image-text similarity matching** using pre-trained CLIP models
- **Mismatch detection** with configurable thresholds
- **Category-based classification** for product images
- **Fine-tuning capabilities** for custom datasets
- **Batch processing** for efficient analysis

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU inference is supported)
- At least 4GB RAM (8GB+ recommended for training)

## Installation

### 1. Install Dependencies

The CLIP feature requires additional dependencies beyond the base installation:

```bash
cd backend

# Install all dependencies including CLIP requirements
pip install -r requirements.txt
```

This will install:
- `torch` - PyTorch for deep learning
- `torchvision` - Vision utilities
- `transformers` - Hugging Face transformers (includes CLIP)
- `pillow` - Image processing
- `pandas` - Dataset handling

### 2. Verify Installation

Test that CLIP dependencies are properly installed:

```bash
python -c "from transformers import CLIPModel, CLIPProcessor; print('CLIP installed successfully')"
```

### 3. Download CLIP Model (Optional)

The default CLIP model (`openai/clip-vit-base-patch32`) will be automatically downloaded on first use. To pre-download it:

```python
from transformers import CLIPModel, CLIPProcessor

# This will download and cache the model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model downloaded successfully!")
```

The model (~600MB) will be cached in `~/.cache/huggingface/`.

## Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory (optional):

```bash
# CLIP Model Configuration
CLIP_MODEL_NAME=openai/clip-vit-base-patch32  # Default pre-trained model
CLIP_MODEL_PATH=                               # Path to fine-tuned model (optional)

# Similarity Threshold
CLIP_SIMILARITY_THRESHOLD=0.25                 # Threshold for mismatch detection (0-1)
                                               # Lower = stricter matching

# Batch Processing
CLIP_BATCH_SIZE=8                              # Batch size for inference

# Training Configuration (if fine-tuning)
CLIP_TRAIN_BATCH_SIZE=16                       # Training batch size
CLIP_TRAIN_LEARNING_RATE=5e-6                  # Learning rate
CLIP_TRAIN_EPOCHS=3                            # Number of epochs
CLIP_CHECKPOINT_DIR=./clip_checkpoints         # Checkpoint directory
```

### Configuration Options

#### Similarity Threshold

The `CLIP_SIMILARITY_THRESHOLD` determines when a mismatch is detected:

- **0.1-0.2**: Very strict (low tolerance for semantic differences)
- **0.25** (default): Balanced (catches clear mismatches)
- **0.3-0.4**: Lenient (allows more semantic variation)

Adjust based on your use case:
```python
# In config.py or via environment variable
CLIP_SIMILARITY_THRESHOLD = 0.25  # Default balanced threshold
```

#### Model Selection

**Pre-trained Models (Zero-shot):**
- `openai/clip-vit-base-patch32` (default) - Good balance of speed and accuracy
- `openai/clip-vit-base-patch16` - Higher accuracy, slower
- `openai/clip-vit-large-patch14` - Best accuracy, much slower

**Custom Fine-tuned Models:**
```bash
# Set path to your fine-tuned model
export CLIP_MODEL_PATH=/path/to/fine-tuned-model
```

## GPU vs CPU

### GPU Configuration (Recommended)

For CUDA-enabled GPUs:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CLIP will automatically use GPU if available. You can verify:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### CPU-only Setup

CLIP works on CPU but will be slower:

```bash
# Regular PyTorch installation (CPU-only)
pip install torch torchvision
```

Expected inference times:
- **GPU**: ~50-100ms per image-text pair
- **CPU**: ~500-1000ms per image-text pair

## Default Categories

The system includes 40+ default product categories for zero-shot classification:

```python
DEFAULT_CATEGORIES = [
    "shoes", "sneakers", "boots", "sandals",
    "shirt", "t-shirt", "blouse", "dress", "pants", "jeans",
    "jacket", "coat", "sweater", "hoodie",
    "handbag", "backpack", "purse", "wallet",
    "watch", "sunglasses", "jewelry",
    # ... and more
]
```

You can customize these in `backend/app/config.py` or pass custom categories to the API.

## Starting the Service

Start the backend server with CLIP enabled:

```bash
cd backend
uvicorn app.main:app --reload
```

The CLIP endpoints will be available at:
- `/clip/analyze` - Analyze image-text similarity
- `/clip/classify` - Classify image by category
- `/clip/categories` - Get supported categories
- `/clip/health` - Check CLIP service health

## API Documentation

Access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing the Installation

### Quick Test via API

```bash
# Test CLIP health
curl http://localhost:8000/clip/health

# Test category classification
curl -X POST http://localhost:8000/clip/classify \
  -F "file=@path/to/product.jpg" \
  -F "top_k=5"

# Test mismatch detection
curl -X POST http://localhost:8000/clip/analyze \
  -F "file=@path/to/product.jpg" \
  -F "text=red leather handbag"
```

### Python Test Script

Create `test_clip.py`:

```python
import requests

# Test health
response = requests.get("http://localhost:8000/clip/health")
print("Health:", response.json())

# Test classification
files = {"file": open("product.jpg", "rb")}
response = requests.post("http://localhost:8000/clip/classify", files=files)
print("Categories:", response.json())

# Test mismatch detection
files = {"file": open("product.jpg", "rb")}
data = {"text": "red leather handbag"}
response = requests.post("http://localhost:8000/clip/analyze", files=files, data=data)
print("Similarity:", response.json())
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter GPU OOM errors:

1. Reduce batch size:
   ```bash
   export CLIP_BATCH_SIZE=4
   ```

2. Use a smaller model:
   ```bash
   export CLIP_MODEL_NAME=openai/clip-vit-base-patch32
   ```

3. Fall back to CPU:
   ```python
   # Force CPU in code
   analyzer = CLIPAnalyzer(device='cpu')
   ```

### Slow Inference

- **Use GPU**: Install CUDA-enabled PyTorch
- **Reduce image size**: CLIP resizes to 224x224, so pre-processing isn't needed
- **Batch requests**: Use `/clip/analyze-batch` for multiple pairs

### Model Download Issues

If model download fails:

```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Or download manually
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Import Errors

```bash
# Reinstall transformers
pip uninstall transformers
pip install transformers

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Next Steps

- **[Training Guide](clip-training.md)** - Fine-tune CLIP on your dataset
- **[Inference Guide](clip-inference.md)** - Detailed API usage examples
- **[Integration Guide](clip-integration.md)** - Integrate with existing workflows

## Performance Optimization

### Lazy Loading

The CLIP model is lazily loaded on first request. To pre-load:

```python
from app.services.clip_analyzer import get_clip_analyzer

# Pre-load model at startup
analyzer = get_clip_analyzer()
```

### Model Caching

Use the singleton pattern to avoid reloading:

```python
# The analyzer is cached after first call
analyzer = get_clip_analyzer()  # Loads model
analyzer = get_clip_analyzer()  # Returns cached instance
```

### Batch Processing

For multiple images, use batch endpoints:

```python
# More efficient than individual requests
items = [
    {"image_path": "img1.jpg", "text": "description 1"},
    {"image_path": "img2.jpg", "text": "description 2"},
]
response = requests.post(
    "http://localhost:8000/clip/analyze-batch",
    json={"items": items}
)
```

## Security Considerations

- **File Upload Limits**: Default 10MB per file (configurable in `config.py`)
- **Rate Limiting**: Consider adding rate limiting for production
- **Input Validation**: Text descriptions are limited to 500 characters
- **Model Security**: Only load models from trusted sources

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Review logs in the console output
3. Open an issue on GitHub with error details
