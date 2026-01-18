# API Spec

## POST /upload
Uploads image and runs analysis. Now includes automatic image-text mismatch detection using CLIP model when description is provided.

## GET /analyze/{id}
Returns one analysis result.

## GET /results
Returns all results.

## POST /check-mismatch
Checks for image-text mismatch using CLIP model.

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