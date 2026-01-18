# CLIP Fine-Tuning Dataset Format

## Overview

This document describes the dataset format required for fine-tuning CLIP models on custom image-text pairs for improved image-description mismatch detection.

## CSV File Format

The training and validation datasets should be provided as CSV files with the following columns:

### Required Columns

| Column Name | Type | Description |
|------------|------|-------------|
| `image_path` | string | Path to the image file (can be absolute or relative to dataset root) |
| `text` | string | Product description or text caption for the image |
| `label` | string | Either "match" or "mismatch" indicating if image and text correspond |

### Example CSV

```csv
image_path,text,label
datasets/images/red_shirt.jpg,Red cotton t-shirt with round neck,match
datasets/images/blue_jeans.jpg,Blue denim jeans with slim fit,match
datasets/images/black_shoes.jpg,Black leather formal shoes,match
datasets/images/red_shirt.jpg,Blue denim jeans with slim fit,mismatch
datasets/images/blue_jeans.jpg,Green cotton dress,mismatch
```

## Label Definition

- **match**: The image and text accurately describe the same product
- **mismatch**: The image and text describe different products or have inconsistencies

## Dataset Preparation Guidelines

### 1. Image Requirements

- **Format**: JPEG, PNG, or other common image formats
- **Resolution**: Minimum 224x224 pixels (CLIP's default input size)
- **Quality**: Clear, well-lit product images
- **Content**: Single product per image for best results

### 2. Text Requirements

- **Length**: 5-100 words for optimal results
- **Content**: Accurate product descriptions including:
  - Color
  - Material
  - Type/category
  - Key features
- **Language**: English (or the language your model is trained for)

### 3. Creating Match Pairs

Match pairs should have:
- Accurate descriptions of the image content
- Consistent color references
- Correct product categories
- Relevant attribute descriptions

### 4. Creating Mismatch Pairs

Mismatch pairs can include:
- **Wrong color**: "Red shirt" with image of blue shirt
- **Wrong category**: "Dress" with image of shoes
- **Wrong material**: "Leather" with image of cotton fabric
- **Wrong features**: "Long sleeve" with image of short sleeve

### 5. Dataset Balance

- Aim for roughly equal numbers of match and mismatch examples
- Recommended minimum: 100 pairs for basic fine-tuning
- Recommended for production: 1000+ pairs for robust performance

### 6. Train/Validation Split

- Typical split: 80% training, 20% validation
- Ensure both splits have balanced match/mismatch ratios
- Avoid having same images in both train and validation sets

## Directory Structure

Recommended directory structure:

```
training/
├── datasets/
│   ├── images/
│   │   ├── product1.jpg
│   │   ├── product2.jpg
│   │   └── ...
│   ├── train.csv
│   ├── val.csv
│   └── README.md
└── train_clip.py
```

## Data Collection Tips

### Using Existing Product Data

1. Export product images and descriptions from your database
2. Create match pairs from correct image-description combinations
3. Create mismatch pairs by randomly shuffling descriptions to different images

### Manual Annotation

1. Use annotation tools to pair images with descriptions
2. Have multiple annotators review mismatch pairs for quality
3. Include edge cases and challenging examples

### Augmentation

For better model generalization:
- Vary description phrasing for the same image
- Include synonyms (e.g., "shirt" vs "t-shirt")
- Include different levels of detail
- Add common misspellings or variations in mismatch pairs

## Quality Assurance

Before training:
1. Verify all image paths are valid
2. Check for duplicate entries
3. Ensure text is properly escaped and free of special characters
4. Review label distribution (should be balanced)
5. Manually inspect a sample of pairs for quality

## Example Data Generation Script

```python
import pandas as pd
import os
from pathlib import Path

def create_sample_dataset(image_dir, output_csv):
    """Create a sample dataset from images in a directory."""
    data = []
    
    for img_path in Path(image_dir).glob("*.jpg"):
        # Create a match pair (you would provide actual descriptions)
        data.append({
            "image_path": str(img_path),
            "text": f"Product image {img_path.stem}",
            "label": "match"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created dataset with {len(data)} entries: {output_csv}")

# Usage
create_sample_dataset("datasets/images", "datasets/train.csv")
```

## Validation

After creating your dataset, validate it using:

```python
import pandas as pd
from pathlib import Path

def validate_dataset(csv_path, root_dir=None):
    """Validate dataset CSV file."""
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required = ['image_path', 'text', 'label']
    missing = set(required) - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    
    # Check for missing values
    if df.isnull().any().any():
        print(f"⚠️  Warning: Dataset contains missing values")
    
    # Check image paths
    root = Path(root_dir or Path(csv_path).parent)
    missing_images = []
    for img_path in df['image_path']:
        full_path = root / img_path if not Path(img_path).is_absolute() else Path(img_path)
        if not full_path.exists():
            missing_images.append(img_path)
    
    if missing_images:
        print(f"❌ Missing images ({len(missing_images)}):")
        for img in missing_images[:5]:
            print(f"   - {img}")
        return False
    
    # Check labels
    valid_labels = {'match', 'mismatch'}
    invalid_labels = set(df['label'].str.lower()) - valid_labels
    if invalid_labels:
        print(f"❌ Invalid labels: {invalid_labels}")
        return False
    
    # Statistics
    print(f"✅ Dataset validated successfully!")
    print(f"   Total entries: {len(df)}")
    print(f"   Match: {(df['label'].str.lower() == 'match').sum()}")
    print(f"   Mismatch: {(df['label'].str.lower() == 'mismatch').sum()}")
    
    return True

# Usage
validate_dataset("datasets/train.csv")
```

## Additional Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
- [Image-Text Contrastive Learning](https://openai.com/research/clip)
