# CLIP-Based Image-Description Mismatch Detection

## Overview

This project now includes CLIP (Contrastive Language-Image Pre-training) integration for advanced image-text similarity analysis. CLIP enables more sophisticated detection of mismatches between product images and their descriptions compared to traditional heuristic methods.

## Features

### 1. Image-Text Similarity Scoring
- Compute semantic similarity between images and text descriptions
- Returns a score from 0 to 1 (higher = more similar)
- Uses state-of-the-art vision-language models

### 2. Match/Mismatch Detection
- Automatic detection of inconsistencies between images and descriptions
- Configurable similarity threshold for classification
- Integrated into existing image quality analysis pipeline

### 3. Zero-Shot Classification (Optional)
- Classify images into predefined categories without training
- Useful for product categorization and validation
- Supports custom label sets

### 4. Fine-Tuning Support
- Train CLIP on your own product image-text pairs
- Improve accuracy for domain-specific products
- Includes training scripts and dataset format guidance

## Quick Start

### Installation

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

The following packages will be installed:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `transformers` - Hugging Face transformers library (includes CLIP)
- `pillow` - Image processing
- `datasets` - Dataset utilities
- `scikit-learn` - Machine learning utilities

### Basic Usage

The CLIP service is automatically integrated into the existing image analysis workflow. When you upload an image with a description, CLIP analysis is performed automatically.

#### Using the API

```bash
# Upload image with description
curl -X POST http://localhost:8000/upload \
  -F "file=@product.jpg" \
  -F "description=Red cotton t-shirt with round neck"
```

The response will include CLIP analysis results:
```json
{
  "id": 1,
  "filename": "product.jpg",
  "passed": true,
  "clip_similarity_score": 0.842,
  "clip_match_status": "Match (score: 0.842)",
  "clip_is_match": true,
  ...
}
```

#### Using the CLI Tool

Test a single image-text pair:
```bash
cd backend
python clip_cli.py test --image path/to/image.jpg --text "Product description"
```

## Configuration

Edit `backend/app/config.py` to customize CLIP settings:

```python
# CLIP configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # Model to use
CLIP_SIMILARITY_THRESHOLD = 0.25  # Match/mismatch threshold (0-1)
CLIP_DEVICE = "cpu"  # Use "cuda" for GPU acceleration
CLIP_ZERO_SHOT_LABELS = []  # Optional labels for classification
```

### Key Parameters

#### CLIP_MODEL_NAME
The Hugging Face model identifier for CLIP. Options include:
- `openai/clip-vit-base-patch32` (default) - Balanced speed/accuracy
- `openai/clip-vit-base-patch16` - Higher accuracy, slower
- `openai/clip-vit-large-patch14` - Best accuracy, requires more memory

#### CLIP_SIMILARITY_THRESHOLD
The threshold for determining match vs. mismatch:
- **0.25** (default) - Balanced, catches obvious mismatches
- **0.20** - More lenient, fewer false positives
- **0.30** - More strict, catches subtle mismatches

**Recommended thresholds by use case:**
- E-commerce general: 0.25
- High-precision requirements: 0.30-0.35
- Exploratory/lenient: 0.20

#### CLIP_DEVICE
- `"cpu"` - Run on CPU (slower but works everywhere)
- `"cuda"` - Run on GPU (much faster, requires NVIDIA GPU with CUDA)

To check if CUDA is available:
```python
import torch
print(torch.cuda.is_available())
```

## Zero-Shot Classification

Zero-shot classification allows you to categorize images without training, using predefined labels.

### Setup

1. Configure labels in `config.py`:
```python
CLIP_ZERO_SHOT_LABELS = [
    "clothing",
    "shoes",
    "accessories",
    "electronics",
    "home goods"
]
```

2. Use in code:
```python
from app.services.clip_service import get_clip_service

service = get_clip_service()
results = service.zero_shot_classify("product.jpg")

print(results)
# Output: {'clothing': 0.72, 'shoes': 0.15, 'accessories': 0.08, ...}
```

### Best Practices

- Use 3-10 mutually exclusive labels
- Make labels specific and descriptive
- Test with sample images to verify performance
- Labels should be in the same language as training data

## Fine-Tuning CLIP

Fine-tuning CLIP on your product data can significantly improve accuracy for domain-specific use cases.

### Dataset Preparation

1. Create a CSV file with image-text pairs:
```csv
image_path,text,label
images/shirt1.jpg,Red cotton t-shirt,match
images/shoe1.jpg,Black leather shoes,match
images/shirt1.jpg,Blue denim jeans,mismatch
```

See [`training/datasets/README.md`](training/datasets/README.md) for detailed format specification.

### Training

#### Using CLI Tool

```bash
cd backend

# Basic training
python clip_cli.py train \
  --train-csv training/datasets/train.csv \
  --val-csv training/datasets/val.csv

# Advanced training with custom parameters
python clip_cli.py train \
  --train-csv training/datasets/train.csv \
  --val-csv training/datasets/val.csv \
  --model openai/clip-vit-base-patch32 \
  --output ./my_fine_tuned_clip \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-5
```

#### Using Python Script Directly

```bash
cd backend

python training/train_clip.py \
  --train_csv training/datasets/train.csv \
  --val_csv training/datasets/val.csv \
  --output_dir ./fine_tuned_clip \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 5e-6
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-csv` | Required | Path to training data CSV |
| `--val-csv` | Optional | Path to validation data CSV |
| `--model` | `openai/clip-vit-base-patch32` | Base model to fine-tune |
| `--output` | `./fine_tuned_clip` | Output directory |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 8 | Training batch size |
| `--lr` | 5e-6 | Learning rate |
| `--save-steps` | 100 | Save checkpoint every N steps |
| `--root-dir` | CSV parent dir | Root for relative image paths |

### Using Fine-Tuned Model

After training, update `config.py` to use your fine-tuned model:

```python
CLIP_MODEL_NAME = "./fine_tuned_clip/best_model"
```

Then restart the backend server:
```bash
uvicorn app.main:app --reload
```

## CLI Commands Reference

### Show Configuration
```bash
python clip_cli.py info
```

### Test Single Image
```bash
python clip_cli.py test \
  --image path/to/image.jpg \
  --text "Product description" \
  --threshold 0.25
```

### Batch Testing
```bash
python clip_cli.py batch-test \
  --csv test_data.csv \
  --threshold 0.25 \
  --output results.csv
```

### Training
```bash
python clip_cli.py train \
  --train-csv datasets/train.csv \
  --val-csv datasets/val.csv \
  --epochs 3 \
  --batch-size 8
```

## API Integration

CLIP results are automatically included in the existing API responses:

### Upload Endpoint Response

```json
{
  "id": 1,
  "filename": "product.jpg",
  "width": 1200,
  "height": 1200,
  "passed": true,
  "reason": "OK",
  "description": "Red cotton shirt",
  
  // New CLIP fields
  "clip_similarity_score": 0.842,
  "clip_match_status": "Match (score: 0.842)",
  "clip_is_match": true,
  
  // Other existing fields...
  "blur_score": 150.5,
  "brightness_score": 120.3,
  ...
}
```

### Understanding CLIP Scores

- **0.8-1.0**: Excellent match, very high confidence
- **0.6-0.8**: Good match, description aligns well with image
- **0.4-0.6**: Moderate similarity, some alignment
- **0.2-0.4**: Weak similarity, possible mismatch
- **0.0-0.2**: Poor match, likely mismatch

## Performance Optimization

### GPU Acceleration

For faster inference, use GPU:

1. Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. Update config:
```python
CLIP_DEVICE = "cuda"
```

### Model Selection

Choose model based on your requirements:

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| clip-vit-base-patch32 | Fast | Good | Low |
| clip-vit-base-patch16 | Medium | Better | Medium |
| clip-vit-large-patch14 | Slow | Best | High |

### Batch Processing

For processing many images, use the batch API:

```python
from app.services.clip_service import get_clip_service

service = get_clip_service()
similarities = service.batch_compute_similarity(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    texts=["desc1", "desc2", "desc3"]
)
```

## Troubleshooting

### CLIP Model Not Loading

**Error**: `Failed to load CLIP model`

**Solution**:
1. Check internet connection (first load downloads model)
2. Ensure sufficient disk space (~500MB for base model)
3. Try manually downloading: `python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"`

### Out of Memory

**Error**: `CUDA out of memory` or system memory issues

**Solutions**:
1. Use smaller model (clip-vit-base-patch32)
2. Reduce batch size during training
3. Use CPU instead of GPU
4. Close other applications

### Low Similarity Scores

**Issue**: All similarity scores are low

**Solutions**:
1. Lower threshold (try 0.20 instead of 0.25)
2. Fine-tune on your specific product domain
3. Ensure descriptions are detailed and accurate
4. Check image quality (blur, lighting)

### Training Not Improving

**Issue**: Validation loss not decreasing

**Solutions**:
1. Increase training data (aim for 500+ pairs minimum)
2. Balance match/mismatch ratios
3. Verify data quality (check labels are correct)
4. Try different learning rates (1e-5 or 1e-6)
5. Train for more epochs

## Sample Code

### Basic Integration

```python
from app.services.clip_service import analyze_image_text_match

# Analyze image-text match
result = analyze_image_text_match(
    image_path="product.jpg",
    description="Red cotton t-shirt"
)

print(f"Match: {result['is_match']}")
print(f"Score: {result['similarity_score']:.3f}")
print(f"Status: {result['status']}")
```

### With Zero-Shot Classification

```python
result = analyze_image_text_match(
    image_path="product.jpg",
    description="Red cotton t-shirt",
    use_zero_shot=True,
    zero_shot_labels=["clothing", "shoes", "accessories"]
)

print(f"Match: {result['is_match']}")
print(f"Categories: {result['zero_shot_results']}")
```

### Custom Threshold

```python
result = analyze_image_text_match(
    image_path="product.jpg",
    description="Red cotton t-shirt",
    threshold=0.30  # More strict
)
```

## Dataset Examples

See the [`training/datasets/`](training/datasets/) directory for:
- `example_train.csv` - Sample training data
- `example_val.csv` - Sample validation data
- `README.md` - Complete dataset format documentation

## Best Practices

### For Development
1. Start with default model and threshold
2. Test with your actual product data
3. Fine-tune if accuracy is insufficient
4. Monitor similarity score distribution

### For Production
1. Use GPU for better performance
2. Cache model in memory (singleton pattern used)
3. Monitor API response times
4. Set appropriate threshold based on testing
5. Regularly retrain with new product data

### For Training
1. Use balanced datasets (50% match, 50% mismatch)
2. Include diverse products and descriptions
3. Validate data quality before training
4. Use validation set to prevent overfitting
5. Save multiple checkpoints

## Additional Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Original research paper
- [Hugging Face CLIP Docs](https://huggingface.co/docs/transformers/model_doc/clip) - Model documentation
- [OpenAI CLIP](https://openai.com/research/clip) - Official project page
- [Training Dataset Format](training/datasets/README.md) - Detailed format guide

## Support

For issues or questions:
1. Check this documentation
2. Review troubleshooting section
3. Check existing GitHub issues
4. Open a new issue with details

## License

This CLIP integration uses models from OpenAI, licensed under MIT License. See model cards on Hugging Face for specific terms.
