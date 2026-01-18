# CLIP Training Pipeline

This directory contains scripts for fine-tuning CLIP models on custom image-text mismatch detection datasets.

## Overview

The training pipeline includes:
- **Dataset loader**: Loads image-text pairs from CSV files
- **Training script**: Fine-tunes CLIP with a classification head
- **Inference script**: Standalone script for predictions

## Dataset Format

Create a CSV file with the following columns:

```csv
image_path,text,label
images/product1.jpg,Red leather handbag with gold hardware,1
images/product2.jpg,Blue cotton shirt,0
images/product3.jpg,White ceramic coffee mug,1
```

**Columns:**
- `image_path`: Path to image file (relative to `--image_base_path`)
- `text`: Text description of the image
- `label`: 
  - `1` = Match (image and text are consistent)
  - `0` = Mismatch (image and text are inconsistent)

**Example Dataset Structure:**
```
data/
├── images/
│   ├── product1.jpg
│   ├── product2.jpg
│   └── product3.jpg
├── train.csv
├── val.csv
└── test.csv
```

## Installation

Install additional dependencies:

```bash
pip install torch transformers Pillow datasets tqdm
```

For GPU training, ensure CUDA is installed and available.

## Usage

### 1. Validate Dataset

Before training, validate your dataset:

```bash
python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --validate_data
```

This checks:
- CSV format and required columns
- Image file accessibility
- Label validity (0 or 1)
- Label distribution

### 2. Train Model

#### Basic Training (CPU)

```bash
python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --output_dir ./fine_tuned_clip \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 1e-5
```

#### GPU Training (Recommended)

```bash
python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --output_dir ./fine_tuned_clip \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --device cuda \
  --num_workers 8
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_csv` | Required | Path to training CSV |
| `--val_csv` | None | Path to validation CSV |
| `--image_base_path` | "" | Base path for images |
| `--model_name` | openai/clip-vit-base-patch32 | Base CLIP model |
| `--output_dir` | ./fine_tuned_clip | Output directory |
| `--epochs` | 10 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 1e-5 | Learning rate |
| `--weight_decay` | 0.01 | Weight decay |
| `--num_workers` | 4 | Data loading workers |
| `--device` | Auto | Device (cuda/cpu) |
| `--validate_data` | False | Validate dataset before training |

### 3. Run Inference

#### Single Image

```bash
python inference_clip.py \
  --model ./fine_tuned_clip/best_model \
  --image test_image.jpg \
  --text "Red leather handbag with gold hardware" \
  --threshold 0.25
```

Output:
```
==================================================
CLIP Mismatch Detection Result
==================================================
Image: test_image.jpg
Text: Red leather handbag with gold hardware
Similarity Score: 0.8743
Threshold: 0.25
Prediction: MATCH
Is Match: True
==================================================
```

#### Batch Inference

```bash
python inference_clip.py \
  --model ./fine_tuned_clip/best_model \
  --csv data/test.csv \
  --image_base_path data/ \
  --output predictions.csv \
  --threshold 0.25
```

The output CSV will contain:
```csv
image_path,text,similarity_score,threshold,is_match,prediction
images/test1.jpg,Description text,0.8743,0.25,True,match
images/test2.jpg,Another description,0.1234,0.25,False,mismatch
```

## Training Tips

### 1. Dataset Preparation

- **Balance**: Aim for roughly equal numbers of match (1) and mismatch (0) samples
- **Diversity**: Include diverse products, backgrounds, and descriptions
- **Quality**: Ensure labels are accurate
- **Size**: Minimum 500-1000 samples recommended, 5000+ for best results

### 2. Hyperparameter Tuning

Start with defaults and adjust based on validation performance:

**Learning Rate:**
- Too high (>1e-4): Unstable training, loss spikes
- Too low (<1e-6): Very slow convergence
- Recommended: 1e-5 to 5e-5

**Batch Size:**
- Larger batch sizes (32-64) for stable training
- Smaller batch sizes (8-16) if GPU memory is limited

**Epochs:**
- Monitor validation loss/accuracy
- Stop when validation performance plateaus
- Use early stopping to prevent overfitting

### 3. Threshold Tuning

After training, tune the threshold on a validation set:

```python
# Run inference with different thresholds
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    # Evaluate metrics (precision, recall, F1)
    # Choose threshold that optimizes your metric
```

**Threshold Guidelines:**
- Lower threshold (0.15-0.25): More sensitive, catches more mismatches (fewer false negatives)
- Higher threshold (0.3-0.4): More conservative, fewer false positives
- Default (0.25): Balanced trade-off

### 4. GPU Training

For faster training, use GPU:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Train on GPU
python train_clip.py --device cuda ...
```

**GPU Memory Requirements:**
- CLIP-ViT-Base: ~4GB VRAM (batch_size=32)
- CLIP-ViT-Large: ~8GB VRAM (batch_size=16)

If GPU memory is insufficient:
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use gradient accumulation (modify training script)

## Model Output

After training, the output directory contains:

```
fine_tuned_clip/
├── best_model/                    # Best model based on validation accuracy
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin
│   ├── classification_head.pt
│   └── training_stats.json
├── checkpoint_epoch_1/            # Checkpoint after each epoch
├── checkpoint_epoch_2/
├── ...
└── final_model/                   # Model after final epoch
```

**Using the Fine-tuned Model:**

Set the environment variable to use your fine-tuned model:

```bash
export CLIP_FINE_TUNED_MODEL_PATH=/path/to/fine_tuned_clip/best_model
```

Or update `backend/app/config.py`:

```python
CLIP_FINE_TUNED_MODEL_PATH = "/path/to/fine_tuned_clip/best_model"
```

## Example Workflow

Complete workflow from dataset to deployment:

```bash
# 1. Prepare dataset
# Create train.csv, val.csv with your labeled data

# 2. Validate dataset
python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --validate_data

# 3. Train model
python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --output_dir ./fine_tuned_clip \
  --epochs 10 \
  --batch_size 32 \
  --device cuda

# 4. Test inference
python inference_clip.py \
  --model ./fine_tuned_clip/best_model \
  --csv data/test.csv \
  --image_base_path data/ \
  --output test_results.csv

# 5. Configure backend to use fine-tuned model
export CLIP_FINE_TUNED_MODEL_PATH="$(pwd)/fine_tuned_clip/best_model"

# 6. Start backend server
cd ../
uvicorn app.main:app --reload
```

## Troubleshooting

### Out of Memory Error

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use CPU: `--device cpu` (slower but no memory limit)
- Close other GPU applications

### Image Not Found Error

```
FileNotFoundError: [Errno 2] No such file or directory: 'images/product1.jpg'
```

**Solution:**
- Check `--image_base_path` is correct
- Verify image paths in CSV are relative to base path
- Ensure all images exist: `--validate_data`

### Label Error

```
ValueError: Invalid label
```

**Solution:**
- Ensure labels are 0 (mismatch) or 1 (match)
- Check for missing values or typos in CSV
- Run with `--validate_data` to find problematic rows

### Slow Training

**Solutions:**
- Use GPU: `--device cuda`
- Increase num_workers: `--num_workers 8`
- Use larger batch size if memory allows
- Consider using a smaller base model

## Advanced Usage

### Custom Base Model

Use a different CLIP variant:

```bash
python train_clip.py \
  --model_name openai/clip-vit-large-patch14 \
  --train_csv data/train.csv \
  ...
```

Available CLIP models:
- `openai/clip-vit-base-patch32` (default, fastest)
- `openai/clip-vit-base-patch16` (more accurate)
- `openai/clip-vit-large-patch14` (best accuracy, slower)

### Resume Training

To resume from a checkpoint:

```bash
# Modify train_clip.py to load from checkpoint
# Or use the checkpoint as the base model
python train_clip.py \
  --model_name ./fine_tuned_clip/checkpoint_epoch_5 \
  ...
```

## Performance Metrics

Track these metrics during training:

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease; if increasing, reduce learning rate or add regularization
- **Validation Accuracy**: Should increase; target >85% for good performance
- **Confusion Matrix**: Balance between false positives and false negatives

Use the saved `training_stats.json` to analyze performance:

```python
import json
import matplotlib.pyplot as plt

with open('fine_tuned_clip/best_model/training_stats.json') as f:
    stats = json.load(f)

plt.plot(stats['train_losses'], label='Train Loss')
plt.plot(stats['val_losses'], label='Val Loss')
plt.legend()
plt.savefig('training_curves.png')
```
