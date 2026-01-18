# CLIP Quick Reference

Quick reference for using the CLIP-based image-text mismatch detection feature.

## Installation

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Basic Usage

### Analyze Image-Text Similarity

```bash
curl -X POST http://localhost:8000/clip/analyze \
  -F "file=@product.jpg" \
  -F "text=red leather handbag"
```

### Classify Image by Category

```bash
curl -X POST http://localhost:8000/clip/classify \
  -F "file=@product.jpg" \
  -F "top_k=3"
```

### Python Example

```python
import requests

# Analyze similarity
with open('product.jpg', 'rb') as img:
    response = requests.post(
        'http://localhost:8000/clip/analyze',
        files={'file': img},
        data={'text': 'red leather handbag'}
    )
    result = response.json()
    print(f"Mismatch: {result['is_mismatch']}")
    print(f"Score: {result['similarity_score']:.3f}")
```

## Configuration

### Environment Variables

```bash
# Optional: Use custom model
export CLIP_MODEL_PATH=/path/to/fine-tuned-model

# Optional: Adjust threshold (default: 0.25)
export CLIP_SIMILARITY_THRESHOLD=0.30
```

## Training

### 1. Prepare Dataset

Create `dataset.csv`:
```csv
image_path,text,label
images/product1.jpg,"red handbag",1
images/product1.jpg,"blue shirt",0
```

### 2. Validate Dataset

```bash
cd backend
python scripts/prepare_dataset.py --csv dataset.csv --validate
```

### 3. Train Model

```bash
python scripts/train_clip.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --epochs 5 \
    --batch-size 16 \
    --output-dir ./clip_model
```

### 4. Use Fine-tuned Model

```bash
export CLIP_MODEL_PATH=./clip_model/best_model
uvicorn app.main:app --reload
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /clip/analyze | POST | Analyze image-text similarity |
| /clip/classify | POST | Classify by category |
| /clip/analyze-batch | POST | Batch analysis |
| /clip/categories | GET | List categories |
| /clip/health | GET | Health check |

## Similarity Scores

| Score | Meaning | Action |
|-------|---------|--------|
| 0.4+ | Excellent match | Accept |
| 0.3-0.4 | Good match | Accept |
| 0.25-0.3 | Fair match | Review |
| 0.15-0.25 | Poor match | Flag |
| < 0.15 | Mismatch | Reject |

## Common Thresholds

- **Strict** (0.15-0.20): Low tolerance for differences
- **Balanced** (0.25): Default, good for most cases
- **Lenient** (0.30-0.35): Higher tolerance for variation

## Troubleshooting

### Out of Memory
```bash
export CLIP_BATCH_SIZE=4
```

### Slow Inference
- Use GPU: Install CUDA-enabled PyTorch
- Use smaller model: `openai/clip-vit-base-patch32`

### Model Download Issues
```bash
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

## Documentation

- **Setup**: [docs/clip-setup.md](clip-setup.md)
- **Training**: [docs/clip-training.md](clip-training.md)
- **Inference**: [docs/clip-inference.md](clip-inference.md)
- **Example Dataset**: [docs/examples/sample_dataset.csv](examples/sample_dataset.csv)

## Support

- Check logs in console output
- Review documentation guides
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Verify model loads: `curl http://localhost:8000/clip/health`
