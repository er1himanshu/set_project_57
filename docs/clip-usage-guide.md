# CLIP Image-Description Mismatch Detection

This guide explains how to use CLIP (Contrastive Language-Image Pre-training) for detecting mismatches between product images and their descriptions.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [API Usage](#api-usage)
4. [Training Custom Models](#training-custom-models)
5. [Configuration](#configuration)
6. [Threshold Tuning](#threshold-tuning)
7. [Best Practices](#best-practices)

## Overview

CLIP is a neural network trained on image-text pairs that can understand the relationship between images and text. We use it to:

- Compute similarity scores between product images and descriptions
- Detect when descriptions don't match images
- Provide more accurate consistency checking than heuristic methods

### How It Works

1. **Image Encoding**: CLIP converts the image into a vector representation
2. **Text Encoding**: CLIP converts the description into a vector representation
3. **Similarity**: Computes cosine similarity between vectors (0-1 scale)
4. **Decision**: Compares similarity to threshold to determine match/mismatch

**Similarity Score Interpretation:**
- **0.8-1.0**: Very strong match
- **0.6-0.8**: Good match
- **0.4-0.6**: Moderate match
- **0.2-0.4**: Weak match (likely mismatch)
- **0.0-0.2**: Very weak match (definite mismatch)

## Quick Start

### Installation

```bash
# Install dependencies
cd backend
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file or set environment variables:

```bash
# Optional: Specify custom model path
export CLIP_FINE_TUNED_MODEL_PATH=/path/to/fine_tuned_model

# Optional: Adjust threshold (default: 0.6)
export CLIP_SIMILARITY_THRESHOLD=0.7

# Optional: Force CPU or use GPU
export CLIP_DEVICE=cuda  # or 'cpu'
```

### Start the Server

```bash
uvicorn app.main:app --reload
```

The CLIP service will automatically load on first use.

## API Usage

### Endpoint 1: Check Mismatch (Upload)

Check mismatch by uploading an image.

**Endpoint:** `POST /clip/check-mismatch`

**Request:**
```bash
curl -X POST "http://localhost:8000/clip/check-mismatch" \
  -F "file=@product.jpg" \
  -F "description=Red leather handbag with gold hardware" \
  -F "threshold=0.25"
```

**Python Example:**
```python
import requests

files = {'file': open('product.jpg', 'rb')}
data = {
    'description': 'Red leather handbag with gold hardware',
    'threshold': 0.3  # Optional, uses config default if omitted
}

response = requests.post(
    'http://localhost:8000/clip/check-mismatch',
    files=files,
    data=data
)

result = response.json()
print(f"Is Match: {result['is_match']}")
print(f"Similarity Score: {result['similarity_score']}")
print(f"Decision: {result['decision']}")
```

**Response:**
```json
{
  "is_match": true,
  "similarity_score": 0.87,
  "decision": "Match (score: 0.870 >= 0.250)",
  "threshold_used": 0.25
}
```

### Endpoint 2: Check Mismatch (By Path)

Check mismatch for already-uploaded images.

**Endpoint:** `POST /clip/check-mismatch-by-path`

**Request:**
```bash
curl -X POST "http://localhost:8000/clip/check-mismatch-by-path" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/uploaded/image.jpg",
    "description": "Blue cotton t-shirt",
    "threshold": 0.25
  }'
```

**Python Example:**
```python
import requests

data = {
    'image_path': '/path/to/uploaded/image.jpg',
    'description': 'Blue cotton t-shirt',
    'threshold': 0.25
}

response = requests.post(
    'http://localhost:8000/clip/check-mismatch-by-path',
    json=data
)

print(response.json())
```

### Endpoint 3: Regular Upload (with CLIP)

The regular upload endpoint now includes CLIP scores:

**Endpoint:** `POST /upload`

**Response includes CLIP fields:**
```json
{
  "result_id": 1,
  "passed": true,
  "description_consistency": "CLIP detected match (score: 0.870)",
  "clip_similarity_score": 0.87,
  "clip_mismatch": false
}
```

### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Training Custom Models

For domain-specific accuracy (e.g., ecommerce products), fine-tune CLIP on your data.

### Dataset Preparation

Create a labeled dataset in CSV format:

**train.csv:**
```csv
image_path,text,label
products/handbag1.jpg,Red leather handbag with gold hardware,1
products/handbag2.jpg,Blue cotton t-shirt,0
products/tshirt1.jpg,White cotton t-shirt,1
products/mug1.jpg,Ceramic coffee mug,1
products/mug2.jpg,Leather briefcase,0
```

**Labels:**
- `1` = Match (image and text are consistent)
- `0` = Mismatch (image and text don't match)

**Tips:**
- Create balanced dataset (50% match, 50% mismatch)
- Include diverse products and descriptions
- 1000+ samples recommended for good results
- Split into train (80%), validation (10%), test (10%)

### Training

See [training/README.md](../backend/training/README.md) for detailed instructions.

**Quick Training:**
```bash
cd backend/training

# Train on GPU (recommended)
python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --output_dir ./fine_tuned_clip \
  --epochs 10 \
  --batch_size 32 \
  --device cuda
```

**Training Time Estimates:**
- 1000 samples, 10 epochs, GPU: ~15-30 minutes
- 5000 samples, 10 epochs, GPU: ~1-2 hours
- CPU training: 5-10x slower

### Using Fine-tuned Models

After training, configure the backend to use your model:

```bash
# Set environment variable
export CLIP_FINE_TUNED_MODEL_PATH=/path/to/fine_tuned_clip/best_model

# Or update config.py
# CLIP_FINE_TUNED_MODEL_PATH = "/path/to/fine_tuned_clip/best_model"

# Restart server
uvicorn app.main:app --reload
```

## Configuration

### Environment Variables

Configure CLIP behavior via environment variables:

```bash
# Model Selection
export CLIP_MODEL_NAME="openai/clip-vit-base-patch32"  # Pre-trained model
export CLIP_FINE_TUNED_MODEL_PATH="/path/to/model"     # Override with fine-tuned

# Similarity Threshold
export CLIP_SIMILARITY_THRESHOLD=0.25  # Adjust based on your needs

# Device Selection
export CLIP_DEVICE=cuda  # Use GPU
# export CLIP_DEVICE=cpu   # Use CPU

# Cache Directory
export CLIP_CACHE_DIR=/path/to/cache  # Where to store downloaded models
```

### Configuration File

Edit `backend/app/config.py`:

```python
# CLIP Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_FINE_TUNED_MODEL_PATH = None  # Set to model path
CLIP_SIMILARITY_THRESHOLD = 0.6  # Updated default
CLIP_DEVICE = "cpu"  # or "cuda"
CLIP_CACHE_DIR = os.path.join(BASE_DIR, "clip_models")
```

### Available Models

Pre-trained CLIP models from HuggingFace:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `openai/clip-vit-base-patch32` | Fast | Good | Default, production |
| `openai/clip-vit-base-patch16` | Medium | Better | Higher accuracy needed |
| `openai/clip-vit-large-patch14` | Slow | Best | Maximum accuracy |

## Threshold Tuning

The threshold determines when to classify as match vs. mismatch.

### Understanding Thresholds

**Lower Threshold (0.4-0.5):**
- ✅ Catches more mismatches (high recall)
- ❌ More false positives (may flag valid matches)
- Use when: False negatives are costly (missing mismatches is bad)

**Higher Threshold (0.7-0.8):**
- ✅ Fewer false positives (high precision)
- ❌ May miss some mismatches (lower recall)
- Use when: False positives are costly (flagging valid matches is bad)

**Balanced Threshold (0.5-0.7):**
- Default recommendation (0.6)
- Good trade-off for most use cases

### Tuning Process

1. **Collect validation data** with known labels
2. **Test different thresholds** on validation set
3. **Calculate metrics** (precision, recall, F1)
4. **Choose threshold** that optimizes your target metric

**Example Script:**

```python
import requests
import pandas as pd
from sklearn.metrics import classification_report

# Load validation data
val_data = pd.read_csv('val.csv')

# Test different thresholds
thresholds = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8]

for threshold in thresholds:
    predictions = []
    
    for _, row in val_data.iterrows():
        response = requests.post(
            'http://localhost:8000/clip/check-mismatch-by-path',
            json={
                'image_path': row['image_path'],
                'description': row['text'],
                'threshold': threshold
            }
        )
        
        result = response.json()
        predictions.append(1 if result['is_match'] else 0)
    
    # Calculate metrics
    print(f"\nThreshold: {threshold}")
    print(classification_report(val_data['label'], predictions))
```

### Monitoring Performance

After deployment, monitor:

- **False Positive Rate**: Valid matches flagged as mismatches
- **False Negative Rate**: Mismatches not detected
- **User Feedback**: Manual review of flagged items

Adjust threshold based on real-world performance.

## Best Practices

### 1. Model Selection

**Use Pre-trained Model When:**
- You don't have labeled training data
- You need quick deployment
- General-purpose matching is sufficient

**Use Fine-tuned Model When:**
- You have domain-specific data (e.g., fashion, electronics)
- You need higher accuracy
- You have 1000+ labeled samples
- You can invest time in training

### 2. Description Quality

CLIP works best with good descriptions:

**Good Description:**
```
"Red leather handbag with gold chain strap and zipper closure"
```

**Poor Description:**
```
"Product" or "Item"
```

**Tips:**
- Be specific and descriptive
- Include color, material, key features
- Avoid generic terms
- 10-50 words is ideal

### 3. Image Quality

Ensure images are:
- Clear and well-lit
- Product is visible
- Minimal background clutter
- Standard ecommerce format

### 4. Performance Optimization

**For Production:**

```python
# Use GPU if available
CLIP_DEVICE = "cuda"

# Cache model in memory (done automatically)
# First request loads model, subsequent requests are fast

# For batch processing, use standalone inference script
# (more efficient than API for large batches)
```

**For Development/Testing:**

```python
# Use smaller model for faster iteration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# CPU is fine for development
CLIP_DEVICE = "cpu"
```

### 5. Error Handling

Handle errors gracefully:

```python
try:
    response = requests.post(...)
    result = response.json()
    
    if result.get('is_match') is None:
        # CLIP check failed, fallback to heuristic
        print("CLIP unavailable, using fallback method")
    else:
        # Use CLIP result
        print(f"Match: {result['is_match']}")
        
except Exception as e:
    print(f"Error: {e}")
    # Implement fallback logic
```

### 6. Cost Considerations

**Computational Cost:**
- Pre-trained CLIP: ~200-500ms per image (CPU), ~50-100ms (GPU)
- Fine-tuning: Requires GPU, ~1-2 hours for 5000 samples
- Storage: Models are ~350MB (base) to ~1.7GB (large)

**Optimization:**
- Batch similar requests together
- Use GPU for production
- Cache results for identical image-text pairs
- Consider async processing for non-critical paths

## Troubleshooting

### Model Won't Load

**Error:** `Failed to load CLIP model`

**Solutions:**
1. Check internet connection (downloads model on first use)
2. Verify model path if using fine-tuned model
3. Check disk space (models are ~350MB-1.7GB)
4. Try specifying a different model

### Out of Memory

**Error:** `CUDA out of memory` or `RuntimeError: out of memory`

**Solutions:**
1. Use CPU: `CLIP_DEVICE=cpu`
2. Use smaller model: `CLIP_MODEL_NAME=openai/clip-vit-base-patch32`
3. Process images sequentially (not in batch)
4. Reduce image resolution before processing

### Slow Performance

**Issue:** API requests taking >5 seconds

**Solutions:**
1. Use GPU: `CLIP_DEVICE=cuda`
2. Use smaller model (patch32 instead of patch16)
3. First request is slow (model loading), subsequent requests are fast
4. For batch processing, use standalone inference script

### Low Accuracy

**Issue:** Many incorrect predictions

**Solutions:**
1. Fine-tune model on your specific domain
2. Adjust threshold based on validation data
3. Improve description quality (more specific)
4. Check image quality (clear, well-lit)
5. Collect more training data

## Example Integration

### Python Client

```python
import requests
from typing import Tuple, Optional

class CLIPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def check_mismatch(
        self,
        image_path: str,
        description: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Check for mismatch between image and description."""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'description': description}
            
            if threshold is not None:
                data['threshold'] = threshold
            
            response = requests.post(
                f'{self.base_url}/clip/check-mismatch',
                files=files,
                data=data
            )
            response.raise_for_status()
            
            result = response.json()
            return result['is_match'], result['similarity_score']

# Usage
client = CLIPClient()
is_match, score = client.check_mismatch(
    'product.jpg',
    'Red leather handbag with gold hardware'
)

print(f"Match: {is_match}, Score: {score:.3f}")
```

## Support

For issues or questions:
1. Check this documentation
2. Review training/README.md for training-specific questions
3. Check API docs at http://localhost:8000/docs
4. Open an issue on GitHub
