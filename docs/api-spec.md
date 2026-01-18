# API Spec

## POST /upload
Uploads image and runs analysis (including CLIP mismatch detection if description provided).

**Request:**
- `file`: Image file (multipart/form-data)
- `description`: Optional product description (form field)

**Response:**
```json
{
  "message": "Image uploaded and analyzed successfully",
  "result_id": 1,
  "passed": true
}
```

## GET /analyze/{id}
Returns one analysis result.

**Response includes:**
- Basic quality metrics (resolution, blur, brightness, etc.)
- Ecommerce metrics (aspect ratio, background, watermarks)
- CLIP metrics (similarity_score, mismatch detection)

## GET /results
Returns all results.

## POST /clip/check-mismatch
Check for image-description mismatch using CLIP.

**Request:**
- `file`: Image file (multipart/form-data)
- `description`: Text description (form field, required)
- `threshold`: Optional similarity threshold (form field)

**Response:**
```json
{
  "is_match": true,
  "similarity_score": 0.87,
  "decision": "Match (score: 0.870 >= 0.250)",
  "threshold_used": 0.25
}
```

## POST /clip/check-mismatch-by-path
Check mismatch for already-uploaded images.

**Request:**
```json
{
  "image_path": "/path/to/image.jpg",
  "description": "Product description",
  "threshold": 0.25
}
```

**Response:**
```json
{
  "is_match": true,
  "similarity_score": 0.87,
  "decision": "Match (score: 0.870 >= 0.250)",
  "threshold_used": 0.25
}
```

## Interactive Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`