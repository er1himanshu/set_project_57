# CLIP Inference Guide

This guide provides detailed examples and best practices for using CLIP-based image-text mismatch detection in your application.

## API Endpoints

### 1. Analyze Image-Text Similarity

Detect if an image and text description match or mismatch.

**Endpoint:** `POST /clip/analyze`

**Request:**
```bash
curl -X POST http://localhost:8000/clip/analyze \
  -F "file=@product_image.jpg" \
  -F "text=red leather handbag with gold hardware" \
  -F "threshold=0.25"
```

**Response:**
```json
{
  "is_mismatch": false,
  "similarity_score": 0.4523,
  "threshold_used": 0.25,
  "confidence": "high",
  "match_quality": "excellent"
}
```

**Response Fields:**
- `is_mismatch`: `true` if similarity < threshold (mismatch detected)
- `similarity_score`: Cosine similarity (0-1, higher = more similar)
- `threshold_used`: Threshold applied
- `confidence`: `high`, `medium`, or `low` (based on distance from threshold)
- `match_quality`: `excellent`, `good`, `fair`, `poor`, or `mismatch`

### 2. Classify Image by Category

Zero-shot classification against product categories.

**Endpoint:** `POST /clip/classify`

**Request:**
```bash
curl -X POST http://localhost:8000/clip/classify \
  -F "file=@product_image.jpg" \
  -F "categories=handbag,backpack,purse,wallet,shoes" \
  -F "top_k=3"
```

**Response:**
```json
{
  "top_matches": [
    {"category": "handbag", "score": 0.8234},
    {"category": "purse", "score": 0.6891},
    {"category": "wallet", "score": 0.3421}
  ]
}
```

### 3. Get Supported Categories

Retrieve default product categories.

**Endpoint:** `GET /clip/categories`

**Request:**
```bash
curl http://localhost:8000/clip/categories
```

**Response:**
```json
{
  "categories": [
    "shoes", "sneakers", "boots", "sandals",
    "shirt", "t-shirt", "dress", "pants",
    // ... more categories
  ],
  "count": 45
}
```

### 4. Batch Analysis

Analyze multiple image-text pairs efficiently.

**Endpoint:** `POST /clip/analyze-batch`

**Request:**
```bash
curl -X POST http://localhost:8000/clip/analyze-batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"image_path": "/path/to/img1.jpg", "text": "red shoes"},
      {"image_path": "/path/to/img2.jpg", "text": "blue shirt"}
    ]
  }'
```

**Note:** Batch endpoint expects server-side image paths, not file uploads.

### 5. Health Check

Check if CLIP service is running.

**Endpoint:** `GET /clip/health`

**Request:**
```bash
curl http://localhost:8000/clip/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "CLIP analyzer",
  "model_loaded": true,
  "device": "cuda"
}
```

## Python Examples

### Basic Similarity Analysis

```python
import requests

def analyze_similarity(image_path, description):
    """Analyze if image and description match."""
    with open(image_path, 'rb') as img:
        files = {'file': img}
        data = {'text': description}
        
        response = requests.post(
            'http://localhost:8000/clip/analyze',
            files=files,
            data=data
        )
        
    return response.json()

# Example usage
result = analyze_similarity('handbag.jpg', 'red leather handbag')

if result['is_mismatch']:
    print(f"⚠️  Mismatch detected! Similarity: {result['similarity_score']:.3f}")
else:
    print(f"✓ Match confirmed! Quality: {result['match_quality']}")
```

### Category Classification

```python
def classify_image(image_path, custom_categories=None, top_k=5):
    """Classify image against product categories."""
    with open(image_path, 'rb') as img:
        files = {'file': img}
        data = {'top_k': top_k}
        
        if custom_categories:
            data['categories'] = ','.join(custom_categories)
        
        response = requests.post(
            'http://localhost:8000/clip/classify',
            files=files,
            data=data
        )
        
    return response.json()

# Example: Auto-detect category
result = classify_image('product.jpg', top_k=3)
print(f"Top category: {result['top_matches'][0]['category']}")
print(f"Confidence: {result['top_matches'][0]['score']:.2%}")

# Example: Custom categories
categories = ['handbag', 'backpack', 'luggage']
result = classify_image('product.jpg', categories, top_k=2)
for match in result['top_matches']:
    print(f"{match['category']}: {match['score']:.2%}")
```

### Batch Processing

```python
def batch_analyze(image_text_pairs):
    """Analyze multiple image-text pairs."""
    items = [
        {'image_path': img, 'text': txt}
        for img, txt in image_text_pairs
    ]
    
    response = requests.post(
        'http://localhost:8000/clip/analyze-batch',
        json={'items': items}
    )
    
    return response.json()

# Example usage
pairs = [
    ('/path/img1.jpg', 'red shoes'),
    ('/path/img2.jpg', 'blue shirt'),
    ('/path/img3.jpg', 'leather wallet'),
]

results = batch_analyze(pairs)
for i, result in enumerate(results['results']):
    print(f"Pair {i}: {'Match' if not result['is_mismatch'] else 'Mismatch'}")
```

### Integration with Upload Workflow

```python
def upload_and_verify(image_path, description):
    """Upload image and verify description matches."""
    # First: CLIP verification
    clip_result = analyze_similarity(image_path, description)
    
    if clip_result['is_mismatch']:
        return {
            'error': 'Description does not match image',
            'similarity_score': clip_result['similarity_score'],
            'suggestion': 'Please provide an accurate description'
        }
    
    # Then: Standard upload
    with open(image_path, 'rb') as img:
        files = {'file': img}
        data = {'description': description}
        
        response = requests.post(
            'http://localhost:8000/upload',
            files=files,
            data=data
        )
    
    return response.json()

# Example
result = upload_and_verify('handbag.jpg', 'red leather handbag')
print(result)
```

## JavaScript/Frontend Examples

### Fetch API

```javascript
async function analyzeSimilarity(imageFile, description) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('text', description);
  
  const response = await fetch('http://localhost:8000/clip/analyze', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Usage with file input
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('imageInput');
  const description = document.getElementById('descInput').value;
  
  if (fileInput.files.length > 0) {
    const result = await analyzeSimilarity(fileInput.files[0], description);
    
    if (result.is_mismatch) {
      alert(`Mismatch detected! Similarity: ${result.similarity_score.toFixed(3)}`);
    } else {
      console.log('Match quality:', result.match_quality);
    }
  }
});
```

### Axios

```javascript
import axios from 'axios';

async function classifyImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('top_k', 5);
  
  try {
    const response = await axios.post(
      'http://localhost:8000/clip/classify',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    
    return response.data.top_matches;
  } catch (error) {
    console.error('Classification failed:', error);
    return [];
  }
}

// Usage in React component
const handleImageUpload = async (event) => {
  const file = event.target.files[0];
  const categories = await classifyImage(file);
  
  setTopCategories(categories);
  console.log('Top category:', categories[0].category);
};
```

## Use Cases

### 1. Product Listing Validation

Verify product descriptions match uploaded images:

```python
def validate_product_listing(image_path, title, description):
    """Validate that image matches product description."""
    # Check title
    title_result = analyze_similarity(image_path, title)
    
    # Check full description
    desc_result = analyze_similarity(image_path, description)
    
    # Both should match
    if title_result['is_mismatch'] or desc_result['is_mismatch']:
        return {
            'valid': False,
            'title_match': not title_result['is_mismatch'],
            'description_match': not desc_result['is_mismatch'],
            'message': 'Description does not match the image'
        }
    
    return {
        'valid': True,
        'message': 'Product listing is valid'
    }
```

### 2. Automatic Category Suggestion

Suggest categories based on image content:

```python
def suggest_category(image_path, min_confidence=0.5):
    """Suggest product category for an image."""
    result = classify_image(image_path, top_k=1)
    top_match = result['top_matches'][0]
    
    if top_match['score'] >= min_confidence:
        return {
            'category': top_match['category'],
            'confidence': top_match['score'],
            'suggestion': f"We suggest categorizing this as: {top_match['category']}"
        }
    else:
        return {
            'category': None,
            'confidence': top_match['score'],
            'suggestion': 'Unable to confidently determine category'
        }
```

### 3. Quality Control Pipeline

Integrate CLIP into quality checks:

```python
def quality_control_pipeline(image_path, description):
    """Run complete quality control on product listing."""
    checks = {}
    
    # 1. CLIP mismatch detection
    clip_result = analyze_similarity(image_path, description)
    checks['description_match'] = not clip_result['is_mismatch']
    checks['similarity_score'] = clip_result['similarity_score']
    
    # 2. Category validation
    expected_category = extract_category_from_description(description)
    if expected_category:
        cat_result = classify_image(image_path, [expected_category], top_k=1)
        checks['category_match'] = cat_result['top_matches'][0]['score'] > 0.5
    
    # 3. Standard image quality checks
    # ... (existing quality checks)
    
    # Overall pass/fail
    checks['passed'] = all([
        checks['description_match'],
        checks.get('category_match', True),
        # ... other quality checks
    ])
    
    return checks
```

### 4. Content Moderation

Detect inappropriate descriptions:

```python
def moderate_listing(image_path, description):
    """Check if description matches image (anti-spam)."""
    result = analyze_similarity(image_path, description)
    
    # Very low similarity = likely spam or misleading
    if result['similarity_score'] < 0.1:
        return {
            'flagged': True,
            'reason': 'Description appears unrelated to image',
            'action': 'manual_review'
        }
    
    return {'flagged': False}
```

## Interpretation Guide

### Similarity Scores

Understanding what similarity scores mean:

| Score Range | Interpretation | Action |
|------------|----------------|--------|
| 0.4 - 1.0 | Excellent match | Accept |
| 0.3 - 0.4 | Good match | Accept |
| 0.25 - 0.3 | Fair match | Review threshold |
| 0.15 - 0.25 | Poor match | Flag for review |
| 0.0 - 0.15 | Clear mismatch | Reject |

### Confidence Levels

The confidence indicates how reliable the decision is:

- **High confidence**: Score is far from threshold (>0.15 away)
  - Safe to automate decisions
  
- **Medium confidence**: Score is moderately far from threshold (0.05-0.15)
  - Consider manual review for critical decisions
  
- **Low confidence**: Score is close to threshold (<0.05)
  - Manual review recommended

### Match Quality

Qualitative descriptions of the match:

- **Excellent**: Strong semantic alignment (score ≥ 0.4)
- **Good**: Clear relationship (score 0.3-0.4)
- **Fair**: Weak relationship (score 0.25-0.3)
- **Poor**: Minimal relationship (score 0.15-0.25)
- **Mismatch**: No meaningful relationship (score < 0.15)

## Optimization Tips

### 1. Caching Results

Cache CLIP results to avoid redundant computation:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_similarity(image_hash, text):
    """Cache similarity results."""
    # Actual API call here
    return analyze_similarity(image_path, text)

# Compute hash for caching
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Usage
img_hash = get_image_hash('product.jpg')
result = cached_similarity(img_hash, 'description')
```

### 2. Threshold Tuning

Find optimal threshold for your use case:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

# Collect scores and labels
scores = []  # Similarity scores
labels = []  # True labels (1=match, 0=mismatch)

for image, text, true_label in test_data:
    result = analyze_similarity(image, text)
    scores.append(result['similarity_score'])
    labels.append(true_label)

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(labels, scores)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
```

### 3. Batch Processing

For high-throughput scenarios:

```python
# Process in batches instead of one-by-one
def process_batch(image_paths, texts, batch_size=32):
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_imgs = image_paths[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        
        batch_results = batch_analyze(zip(batch_imgs, batch_texts))
        results.extend(batch_results['results'])
    
    return results
```

## Troubleshooting

### Low Similarity Scores

If all similarity scores are unexpectedly low:

1. **Check image quality**: Ensure images are clear and well-lit
2. **Verify descriptions**: Make sure text is detailed and accurate
3. **Consider fine-tuning**: Pre-trained CLIP may need domain adaptation
4. **Adjust threshold**: Default (0.25) may be too high for your domain

### High False Positive Rate

If getting too many false mismatches:

1. **Lower threshold**: Try 0.2 or 0.15
2. **Improve descriptions**: More detailed text improves matching
3. **Use category classification**: Supplement with category checks
4. **Fine-tune model**: Train on your specific products

### Inconsistent Results

If results vary for similar inputs:

1. **Check image preprocessing**: Ensure consistent image quality
2. **Normalize text**: Remove special characters, lowercase
3. **Use cached analyzer**: Don't reinitialize model per request
4. **Batch process**: More stable than individual requests

## Next Steps

- **[Training Guide](clip-training.md)** - Fine-tune for your domain
- **[Setup Guide](clip-setup.md)** - Configuration and deployment
- Integrate CLIP with existing quality checks
- Monitor similarity score distributions
- Collect edge cases for model improvement
