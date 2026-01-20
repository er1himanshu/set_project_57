# API Spec

## POST /upload
Uploads image and runs analysis. Now includes automatic image-text mismatch detection using CLIP model when description is provided.

## GET /analyze/{id}
Returns one analysis result.

## GET /results
Returns all results.

## POST /check-mismatch
Checks for image-text mismatch using CLIP model with category-based detection.

**Parameters:**
- `file`: Image file (required)
- `description`: Product description text (required, min 10 chars)
- `threshold`: Optional similarity threshold (0-1, default: 0.25)

**Response:**
```json
{
  "filename": "unique_filename.jpg",
  "description": "Product description",
  "has_mismatch": false,
  "similarity_score": 0.85,
  "threshold": 0.25,
  "message": "Match confirmed (score: 0.85)",
  "recommendation": "Image and description match well."
}
```

**Enhanced Messaging:**
When a category mismatch is detected, the `message` field includes additional context:
```json
{
  "message": "Mismatch detected (score: 0.18). Description mentions 'bike', but image looks like 'shoes'."
}
```

## POST /explain
Generates CLIP explainability with attention rollout heatmap.

**Parameters:**
- `file`: Image file (required)
- `description`: Product description text (required, min 10 chars)
- `threshold`: Optional similarity threshold (0-1, default: 0.25)

**Response:**
```json
{
  "filename": "unique_filename.jpg",
  "description": "Product description",
  "similarity_score": 0.85,
  "has_mismatch": false,
  "threshold": 0.25,
  "message": "Match confirmed (score: 0.85)",
  "heatmap_base64": "<base64-encoded PNG>",
  "explanation": "Heatmap shows which image regions most influenced..."
}
```

**Error Handling:**
- Returns 400 for validation errors
- Returns 503 when CLIP model is unavailable
- Returns 500 for processing errors
- All errors include descriptive messages to help troubleshoot