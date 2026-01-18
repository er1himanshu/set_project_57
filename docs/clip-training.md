# CLIP Training Guide

This guide covers fine-tuning CLIP models on custom image-text match/mismatch datasets for improved accuracy on your specific product domain.

## Overview

Fine-tuning CLIP allows you to:
- **Improve accuracy** on your specific product types
- **Reduce false positives** in mismatch detection
- **Adapt to domain-specific language** (e.g., fashion vs. electronics terminology)
- **Optimize for your threshold preferences**

## Dataset Format

### CSV Structure

Your dataset should be a CSV file with three required columns:

```csv
image_path,text,label
/path/to/images/product_001.jpg,"red leather handbag with gold hardware",1
/path/to/images/product_001.jpg,"blue cotton t-shirt",0
/path/to/images/product_002.jpg,"wireless bluetooth headphones",1
/path/to/images/product_002.jpg,"wooden dining table",0
```

**Column Definitions:**

1. **`image_path`** (string): Relative or absolute path to the image file
   - Can be relative to a base path specified during training
   - Supports: `.jpg`, `.jpeg`, `.png`, `.webp`

2. **`text`** (string): Product description or caption
   - Natural language description
   - Can be short phrases or detailed descriptions
   - Typically 5-100 words

3. **`label`** (integer): Match/mismatch label
   - `1` = Match (image and text correspond)
   - `0` = Mismatch (image and text don't match)

### Example Dataset

Here's a complete example with proper matches and mismatches:

```csv
image_path,text,label
data/images/shoes_001.jpg,"black leather oxford dress shoes",1
data/images/shoes_001.jpg,"red running sneakers",0
data/images/shoes_001.jpg,"silver laptop computer",0
data/images/laptop_001.jpg,"15 inch silver laptop with backlit keyboard",1
data/images/laptop_001.jpg,"white ceramic coffee mug",0
data/images/handbag_001.jpg,"brown leather tote bag with shoulder strap",1
data/images/handbag_001.jpg,"blue denim jeans",0
```

### Dataset Requirements

**Size:**
- Minimum: 100 pairs (50 match, 50 mismatch)
- Recommended: 1000+ pairs (500+ each)
- Ideal: 10,000+ pairs for production use

**Balance:**
- Maintain ~50/50 ratio of match/mismatch samples
- The training script will warn if ratio is > 5:1 or < 1:5

**Quality:**
- Clear, high-quality images (same as inference)
- Accurate, descriptive text
- Diverse product types and descriptions
- Verified labels (incorrect labels hurt performance)

## Creating Your Dataset

### 1. Collect Images and Descriptions

Gather product images and their true descriptions:

```python
import pandas as pd
import os

data = []

# Example: Match pairs
products = [
    ("products/shoe_1.jpg", "black leather dress shoes", 1),
    ("products/bag_1.jpg", "brown leather handbag", 1),
    # ... more matches
]

# Add to dataset
for img_path, text, label in products:
    if os.path.exists(img_path):
        data.append({
            'image_path': img_path,
            'text': text,
            'label': label
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)
```

### 2. Generate Mismatch Pairs

For mismatch examples, pair images with incorrect descriptions:

```python
import random

matches = df[df['label'] == 1].copy()

# Create mismatches by shuffling descriptions
mismatches = []
for idx, row in matches.iterrows():
    # Get a different random description
    other_texts = matches[matches['image_path'] != row['image_path']]['text'].tolist()
    if other_texts:
        mismatch_text = random.choice(other_texts)
        mismatches.append({
            'image_path': row['image_path'],
            'text': mismatch_text,
            'label': 0
        })

# Combine matches and mismatches
full_dataset = pd.concat([matches, pd.DataFrame(mismatches)])
full_dataset = full_dataset.sample(frac=1).reset_index(drop=True)  # Shuffle
full_dataset.to_csv('full_dataset.csv', index=False)
```

### 3. Validate Dataset

Before training, validate your dataset:

```bash
cd backend
python scripts/prepare_dataset.py \
    --csv data/full_dataset.csv \
    --image-base-path /path/to/images \
    --validate
```

This checks:
- All required columns present
- No missing values
- Labels are 0 or 1
- All images exist and are readable
- Dataset balance

### 4. Split into Train/Val Sets

Split your dataset for training and validation:

```bash
python scripts/prepare_dataset.py \
    --csv data/full_dataset.csv \
    --split 0.8 \
    --output-dir data \
    --shuffle
```

This creates:
- `data/train.csv` - 80% of data for training
- `data/val.csv` - 20% of data for validation

## Training Process

### Basic Training

Train with default settings:

```bash
cd backend
python scripts/train_clip.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --output-dir ./clip_checkpoints
```

### Advanced Training Options

Full command with all options:

```bash
python scripts/train_clip.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --image-base-path /absolute/path/to/images \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 5e-6 \
    --model-name openai/clip-vit-base-patch32 \
    --output-dir ./clip_checkpoints \
    --save-all-checkpoints \
    --device cuda
```

**Parameter Guide:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-csv` | Required | Path to training CSV |
| `--val-csv` | None | Path to validation CSV (recommended) |
| `--image-base-path` | None | Base path prepended to image_path in CSV |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 5e-6 | Learning rate (Adam optimizer) |
| `--model-name` | clip-vit-base-patch32 | Base model to fine-tune |
| `--output-dir` | ./clip_checkpoints | Output directory |
| `--save-all-checkpoints` | False | Save every epoch (default: best only) |
| `--device` | auto | Force 'cuda' or 'cpu' |

### Training Output

The script outputs:

```
==============================================================
CLIP Fine-tuning Script
==============================================================

Validating training dataset...
Training dataset: data/train.csv
  Total samples: 800
  Match samples: 400
  Mismatch samples: 400
✓ Training dataset is valid

Initializing CLIP trainer...
  Base model: openai/clip-vit-base-patch32
  Learning rate: 5e-06
  Device: cuda

Starting training...
  Epochs: 3
  Batch size: 16
  Output directory: ./clip_checkpoints

Epoch 1/3
Training: 100%|████████| 50/50 [01:23<00:00, 1.67s/it, loss=0.245]
Training loss: 0.2456
Validation: 100%|████████| 13/13 [00:15<00:00, 1.19s/it, loss=0.198]
Validation loss: 0.1984
Saved best model with val_loss: 0.1984

...

==============================================================
Training Complete!
==============================================================

Final Training Loss: 0.1234
Final Validation Loss: 0.1456
Best Validation Loss: 0.1456

Model saved to: ./clip_checkpoints
```

### Checkpoints

Training creates these checkpoints:

```
clip_checkpoints/
├── best_model/              # Best validation loss model
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── checkpoint_epoch_1/      # If --save-all-checkpoints
├── checkpoint_epoch_2/
└── final_model/             # Final epoch model
```

## Using Fine-tuned Models

### Configure Model Path

Set the environment variable to use your fine-tuned model:

```bash
export CLIP_MODEL_PATH=/path/to/clip_checkpoints/best_model
```

Or in your `.env` file:

```
CLIP_MODEL_PATH=/path/to/clip_checkpoints/best_model
```

### Test Fine-tuned Model

```python
import requests

# The API will automatically use the fine-tuned model
files = {"file": open("test_product.jpg", "rb")}
data = {"text": "red leather handbag"}
response = requests.post(
    "http://localhost:8000/clip/analyze",
    files=files,
    data=data
)
print(response.json())
```

### Compare Models

Test both pre-trained and fine-tuned:

```python
from app.services.clip_analyzer import CLIPAnalyzer

# Pre-trained
pretrained = CLIPAnalyzer(model_path=None)
score1 = pretrained.compute_similarity("image.jpg", "description")

# Fine-tuned
finetuned = CLIPAnalyzer(model_path="./clip_checkpoints/best_model")
score2 = finetuned.compute_similarity("image.jpg", "description")

print(f"Pre-trained: {score1:.3f}")
print(f"Fine-tuned: {score2:.3f}")
```

## Training Tips

### Hyperparameter Tuning

**Learning Rate:**
- Too high (>1e-5): Training unstable, may diverge
- Too low (<1e-7): Training too slow
- Recommended: 5e-6 to 1e-5

**Batch Size:**
- Larger (32+): Faster training, needs more GPU memory
- Smaller (8-16): Works on smaller GPUs
- Adjust based on available VRAM

**Epochs:**
- Start with 3-5 epochs
- Monitor validation loss for overfitting
- Stop if validation loss increases

### Data Augmentation

For small datasets, augment with:

```python
from PIL import Image, ImageEnhance
import random

def augment_image(image_path):
    img = Image.open(image_path)
    
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    return img
```

### Monitoring Training

Watch for these signs:

**Good Training:**
- Training loss decreases steadily
- Validation loss decreases similarly
- Gap between train/val loss is small

**Overfitting:**
- Training loss much lower than validation loss
- Validation loss stops decreasing or increases
- Solution: More data, fewer epochs, regularization

**Underfitting:**
- Both losses remain high
- Solution: More epochs, higher learning rate, larger model

## Advanced Topics

### Multi-GPU Training

For faster training on multiple GPUs:

```python
# Modify clip_trainer.py to use DataParallel
import torch.nn as nn

model = nn.DataParallel(model)
```

### Continue Training

Resume from a checkpoint:

```bash
python scripts/train_clip.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --model-name ./clip_checkpoints/checkpoint_epoch_2 \
    --epochs 5 \
    --output-dir ./clip_checkpoints_continued
```

### Custom Loss Functions

Modify the training loss in `clip_trainer.py`:

```python
# Current: Binary cross-entropy
loss = torch.nn.functional.binary_cross_entropy(similarity, labels)

# Alternative: Contrastive loss, triplet loss, etc.
```

## Evaluation

### Compute Metrics

After training, evaluate on test set:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Load test data
test_df = pd.read_csv('data/test.csv')

# Run inference
from app.services.clip_analyzer import get_clip_analyzer
analyzer = get_clip_analyzer(model_path='./clip_checkpoints/best_model')

predictions = []
for _, row in test_df.iterrows():
    result = analyzer.detect_mismatch(row['image_path'], row['text'])
    predictions.append(0 if result['is_mismatch'] else 1)

# Calculate metrics
accuracy = accuracy_score(test_df['label'], predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_df['label'], predictions, average='binary'
)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_clip.py ... --batch-size 8

# Or use CPU
python scripts/train_clip.py ... --device cpu
```

### Dataset Validation Errors

```bash
# Check what's wrong
python scripts/prepare_dataset.py --csv data/train.csv --validate

# Common issues:
# - Missing columns: Add required columns
# - Missing images: Fix paths or --image-base-path
# - Invalid labels: Must be 0 or 1
```

### Training Doesn't Converge

- Check dataset quality and labels
- Reduce learning rate: `--learning-rate 1e-6`
- Increase epochs: `--epochs 10`
- Try different base model

## Example Workflow

Complete workflow from data to deployment:

```bash
# 1. Create sample dataset structure
python scripts/prepare_dataset.py --create-sample 100 --csv data/sample.csv

# 2. Replace with your actual data
# ... edit data/sample.csv with real image paths and descriptions

# 3. Validate dataset
python scripts/prepare_dataset.py --csv data/dataset.csv --validate

# 4. Split into train/val
python scripts/prepare_dataset.py \
    --csv data/dataset.csv \
    --split 0.8 \
    --output-dir data \
    --shuffle

# 5. Train model
python scripts/train_clip.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --epochs 5 \
    --batch-size 16 \
    --output-dir ./clip_model

# 6. Configure to use fine-tuned model
export CLIP_MODEL_PATH=./clip_model/best_model

# 7. Start server
uvicorn app.main:app --reload

# 8. Test
curl -X POST http://localhost:8000/clip/analyze \
    -F "file=@test.jpg" \
    -F "text=test description"
```

## Next Steps

- **[Inference Guide](clip-inference.md)** - Use your fine-tuned model
- **[Setup Guide](clip-setup.md)** - Configuration and deployment
- Experiment with different thresholds
- Collect more training data for continuous improvement
