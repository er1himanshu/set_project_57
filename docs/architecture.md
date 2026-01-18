# Architecture

## Backend Stack
- FastAPI backend
- OpenCV + Scikit-image for image quality analysis
- CLIP (via Hugging Face Transformers) for semantic image-text matching
- PyTorch for deep learning inference and training
- SQLite database

## Frontend Stack
- React + Tailwind frontend

## CLIP Integration

### Components

1. **clip_analyzer.py**: Zero-shot inference engine
   - Image-text similarity computation
   - Mismatch detection with configurable thresholds
   - Category-based classification
   - Batch processing support

2. **clip_trainer.py**: Fine-tuning utilities
   - Custom dataset loading (CSV format)
   - Training loop with validation
   - Checkpoint management
   - Dataset validation tools

3. **clip.py (routes)**: REST API endpoints
   - `/clip/analyze` - Similarity analysis
   - `/clip/classify` - Category classification
   - `/clip/analyze-batch` - Batch processing
   - `/clip/categories` - Category listing
   - `/clip/health` - Health check

### Data Flow

```
User Upload → FastAPI → CLIP Analyzer → Response
                ↓
         Image Quality Check → Database
```

### Training Flow

```
Dataset CSV → Validation → Training → Checkpoints → Deployment
```

### Models

- **Pre-trained**: openai/clip-vit-base-patch32 (default)
- **Fine-tuned**: Custom models trained on domain-specific data
- **Storage**: Local checkpoint directory or HuggingFace cache