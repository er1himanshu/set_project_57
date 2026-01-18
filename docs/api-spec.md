# API Spec

## Standard Image Analysis Endpoints

### POST /upload
Uploads image and runs analysis.

### GET /analyze/{id}
Returns one analysis result.

### GET /results
Returns all results.

## CLIP Analysis Endpoints

### POST /clip/analyze
Analyze image-text similarity and detect mismatches.

**Request:**
- `file`: Image file (multipart/form-data)
- `text`: Description to compare (form field)
- `threshold`: Optional similarity threshold (form field, default: 0.25)

**Response:**
```json
{
  "is_mismatch": false,
  "similarity_score": 0.45,
  "threshold_used": 0.25,
  "confidence": "high",
  "match_quality": "excellent"
}
```

### POST /clip/classify
Classify image against product categories (zero-shot).

**Request:**
- `file`: Image file (multipart/form-data)
- `categories`: Optional comma-separated categories (form field)
- `top_k`: Number of results to return (form field, default: 5)

**Response:**
```json
{
  "top_matches": [
    {"category": "handbag", "score": 0.82},
    {"category": "purse", "score": 0.69}
  ]
}
```

### POST /clip/analyze-batch
Batch analyze multiple image-text pairs.

**Request:**
```json
{
  "items": [
    {"image_path": "/path/img1.jpg", "text": "description 1"},
    {"image_path": "/path/img2.jpg", "text": "description 2"}
  ]
}
```

**Response:**
```json
{
  "results": [...],
  "errors": [...]
}
```

### GET /clip/categories
Get list of supported product categories.

**Response:**
```json
{
  "categories": ["shoes", "handbag", "laptop", ...],
  "count": 45
}
```

### GET /clip/config
Get current CLIP configuration.

**Response:**
```json
{
  "similarity_threshold": 0.25,
  "default_categories_count": 45,
  "model_loaded": true
}
```

### GET /clip/health
Health check for CLIP service.

**Response:**
```json
{
  "status": "healthy",
  "service": "CLIP analyzer",
  "model_loaded": true,
  "device": "cuda"
}
```