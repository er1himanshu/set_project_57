# AI-Powered Ecommerce Product Listing Evaluator

A full-stack application for analyzing product images with comprehensive AI-powered quality metrics specifically designed for ecommerce marketplaces. Features advanced CLIP-based image-text mismatch detection to ensure product descriptions accurately match images.

## 🌟 Features

### Image Quality Analysis
- **Resolution Validation**: Ensures images meet minimum 1000×1000 pixel requirement
- **Blur Detection**: Uses Laplacian variance to detect unfocused or blurry images
- **Sharpness Analysis**: Edge detection to verify image clarity
- **Brightness & Contrast**: Validates proper lighting conditions
- **Aspect Ratio Check**: Validates standard ecommerce ratios (1:1, 4:3, 16:9, etc.)

### Ecommerce Standards
- **Background Assessment**: Scores background cleanliness for white/neutral backgrounds
- **Watermark Detection**: Identifies text overlays and watermarks
- **Description Consistency**: Validates alignment between product description and image content (color matching, basic heuristics)

### 🆕 CLIP-Powered Mismatch Detection
- **Semantic Similarity Analysis**: Deep learning-based image-text matching using OpenAI CLIP
- **Zero-shot Classification**: Automatic product category detection from images
- **Configurable Thresholds**: Customizable mismatch detection sensitivity
- **Fine-tuning Support**: Train custom models on your product domain
- **Batch Processing**: Efficient analysis of multiple image-text pairs
- **API Integration**: RESTful endpoints for seamless integration

### User Experience
- **Modern UI**: Beautiful gradient-based design with card layouts
- **Image Preview**: See your image before upload
- **Real-time Results**: Instant analysis with detailed quality checklist
- **Improvement Suggestions**: Actionable tips to enhance image quality
- **Results Dashboard**: Track all analyses with statistics and filters

## 🚀 Prerequisites

- **Python** 3.8 or higher
- **Node.js** 16 or higher
- **pip** (Python package manager)
- **npm** (Node package manager)
- **Optional**: CUDA-capable GPU for faster CLIP inference

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/er1himanshu/set_project_57.git
cd set_project_57
```

### 2. Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment (recommended)
python3 -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --reload
```

The backend API will run at **http://localhost:8000**
- Interactive API documentation: **http://localhost:8000/docs**
- Alternative API docs: **http://localhost:8000/redoc**

### 3. Frontend Setup (React + Vite)

Open a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will run at **http://localhost:5173**

## 🎯 Usage

### Uploading Images for Analysis

1. Navigate to **http://localhost:5173** in your web browser
2. Click the upload area or drag and drop a product image
3. (Optional) Enter a product description for consistency checking
4. Click "Analyze Image Quality"
5. View detailed results with pass/fail status, quality metrics, and improvement suggestions

### Viewing Results

- Click "Results" in the navigation bar to see all analyzed images
- View statistics: total analyzed, passed, and failed counts
- Expand suggestions for failed images to see improvement tips

## 📡 API Endpoints

### Standard Image Analysis

#### Upload Image
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - description: product description text (optional)

Response:
{
  "message": "Uploaded & analyzed",
  "result_id": 1
}
```

#### Get All Results
```http
GET /results

Response: Array of ImageResultSchema objects
```

#### Get Specific Result
```http
GET /results/{result_id}

Response: ImageResultSchema object with all quality metrics
```

#### Get Analysis by Image ID
```http
GET /analyze/{image_id}

Response: ImageResultSchema object
```

### 🆕 CLIP Analysis Endpoints

#### Analyze Image-Text Similarity
```http
POST /clip/analyze
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - text: description to compare (required)
  - threshold: similarity threshold (optional, default: 0.25)

Response:
{
  "is_mismatch": false,
  "similarity_score": 0.45,
  "threshold_used": 0.25,
  "confidence": "high",
  "match_quality": "excellent"
}
```

#### Classify Image by Category
```http
POST /clip/classify
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - categories: comma-separated category list (optional)
  - top_k: number of results (optional, default: 5)

Response:
{
  "top_matches": [
    {"category": "handbag", "score": 0.82},
    {"category": "purse", "score": 0.69}
  ]
}
```

#### Get Supported Categories
```http
GET /clip/categories

Response:
{
  "categories": ["shoes", "handbag", "laptop", ...],
  "count": 45
}
```

#### CLIP Health Check
```http
GET /clip/health

Response:
{
  "status": "healthy",
  "service": "CLIP analyzer",
  "model_loaded": true,
  "device": "cuda"
}
```

## 📊 Quality Criteria

Images are evaluated against the following ecommerce standards:

| Metric | Requirement | Description |
|--------|-------------|-------------|
| **Resolution** | ≥ 1000×1000 px | Minimum pixel dimensions |
| **Blur Score** | ≥ 100.0 | Laplacian variance threshold |
| **Sharpness** | ≥ 50.0 | Edge detection score |
| **Brightness** | 60-200 | Optimal lighting range |
| **Aspect Ratio** | Standard ratios | 1:1, 4:3, 3:4, 16:9, 9:16 (±10% tolerance) |
| **Background** | ≥ 70% | Clean/white background score |
| **Watermarks** | None | No text overlays or watermarks |
| **Description** | Consistent | Color and content matching |
| **CLIP Similarity** | ≥ 0.25 | Semantic image-text match (CLIP-based) |

### Response Schema

```json
{
  "id": 1,
  "filename": "product.jpg",
  "width": 1200,
  "height": 1200,
  "blur_score": 150.5,
  "brightness_score": 120.3,
  "contrast_score": 45.2,
  "passed": true,
  "reason": "OK",
  "description": "Red leather handbag",
  "aspect_ratio": 1.0,
  "sharpness_score": 85.7,
  "background_score": 0.85,
  "has_watermark": false,
  "description_consistency": "Consistent",
  "improvement_suggestions": "Image meets quality standards"
}
```

## 🧠 CLIP-Based Mismatch Detection

### What is CLIP?

CLIP (Contrastive Language-Image Pre-training) is a neural network trained on millions of image-text pairs. It understands semantic relationships between images and text, enabling:

- **Zero-shot classification**: Identify product categories without specific training
- **Semantic matching**: Detect if descriptions actually match images
- **Beyond keywords**: Understands meaning, not just word matching

### Quick Start with CLIP

#### 1. Basic Usage

```bash
# Analyze image-text similarity
curl -X POST http://localhost:8000/clip/analyze \
  -F "file=@product.jpg" \
  -F "text=red leather handbag"

# Response:
# {
#   "is_mismatch": false,
#   "similarity_score": 0.45,
#   "confidence": "high",
#   "match_quality": "excellent"
# }
```

#### 2. Category Classification

```bash
# Auto-detect product category
curl -X POST http://localhost:8000/clip/classify \
  -F "file=@product.jpg" \
  -F "top_k=3"

# Response:
# {
#   "top_matches": [
#     {"category": "handbag", "score": 0.82},
#     {"category": "purse", "score": 0.69},
#     {"category": "wallet", "score": 0.34}
#   ]
# }
```

#### 3. Python Integration

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
    
    if result['is_mismatch']:
        print(f"⚠️ Mismatch detected! Score: {result['similarity_score']:.3f}")
    else:
        print(f"✓ Match confirmed! Quality: {result['match_quality']}")
```

### Configuration

Configure CLIP behavior via environment variables:

```bash
# In backend/.env or export
CLIP_MODEL_NAME=openai/clip-vit-base-patch32  # Default model
CLIP_SIMILARITY_THRESHOLD=0.25                 # Mismatch threshold (0-1)
CLIP_MODEL_PATH=/path/to/finetuned/model      # Optional: fine-tuned model
```

**Threshold Guide:**
- **0.15-0.20**: Strict (low tolerance for variation)
- **0.25** (default): Balanced
- **0.30-0.35**: Lenient (allows more semantic flexibility)

### Fine-tuning CLIP

For improved accuracy on your specific products, fine-tune CLIP on custom data:

#### 1. Prepare Dataset

Create CSV with image paths, descriptions, and labels:

```csv
image_path,text,label
products/handbag_1.jpg,"red leather handbag",1
products/handbag_1.jpg,"blue cotton t-shirt",0
products/laptop_1.jpg,"15 inch silver laptop",1
products/laptop_1.jpg,"wooden dining chair",0
```

Where `label`: 1 = match, 0 = mismatch

See full example: [`docs/examples/sample_dataset.csv`](docs/examples/sample_dataset.csv)

#### 2. Validate Dataset

```bash
cd backend
python scripts/prepare_dataset.py \
    --csv data/dataset.csv \
    --validate
```

#### 3. Train Model

```bash
python scripts/train_clip.py \
    --train-csv data/train.csv \
    --val-csv data/val.csv \
    --epochs 5 \
    --batch-size 16 \
    --output-dir ./clip_checkpoints
```

#### 4. Use Fine-tuned Model

```bash
export CLIP_MODEL_PATH=./clip_checkpoints/best_model
uvicorn app.main:app --reload
```

### Documentation

Comprehensive CLIP guides available in [`docs/`](docs/):

- **[CLIP Setup Guide](docs/clip-setup.md)** - Installation, configuration, and deployment
- **[CLIP Training Guide](docs/clip-training.md)** - Fine-tuning on custom datasets
- **[CLIP Inference Guide](docs/clip-inference.md)** - API usage and integration examples

## 🏗️ Project Structure

```
set_project_57/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── models.py            # SQLAlchemy database models
│   │   ├── schemas.py           # Pydantic validation schemas
│   │   ├── config.py            # Configuration and thresholds
│   │   ├── database.py          # Database setup and connection
│   │   ├── routes/
│   │   │   ├── upload.py        # Image upload endpoint
│   │   │   ├── analyze.py       # Analysis endpoint
│   │   │   ├── results.py       # Results retrieval endpoints
│   │   │   └── clip.py          # 🆕 CLIP analysis endpoints
│   │   ├── services/
│   │   │   ├── image_quality.py # Image analysis logic
│   │   │   ├── storage.py       # File storage management
│   │   │   ├── clip_analyzer.py # 🆕 CLIP inference utilities
│   │   │   └── clip_trainer.py  # 🆕 CLIP fine-tuning utilities
│   │   └── utils/
│   │       └── helpers.py       # Utility functions
│   ├── scripts/
│   │   ├── train_clip.py        # 🆕 CLIP training script
│   │   └── prepare_dataset.py   # 🆕 Dataset preparation tool
│   ├── requirements.txt         # Python dependencies
│   └── uploads/                 # Uploaded images (auto-created)
├── docs/
│   ├── clip-setup.md            # 🆕 CLIP setup guide
│   ├── clip-training.md         # 🆕 CLIP training guide
│   ├── clip-inference.md        # 🆕 CLIP inference guide
│   ├── examples/
│   │   └── sample_dataset.csv   # 🆕 Example training dataset
│   ├── architecture.md          # Architecture overview
│   └── api-spec.md              # API specification
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx    # Home page with upload
│   │   │   └── Results.jsx      # Results listing page
│   │   ├── components/
│   │   │   ├── Navbar.jsx       # Navigation component
│   │   │   ├── UploadForm.jsx   # Upload and analysis UI
│   │   │   └── ResultsTable.jsx # Results display table
│   │   ├── api/
│   │   │   └── client.js        # API client
│   │   ├── App.jsx              # Main app component
│   │   └── main.jsx             # Entry point
│   ├── index.html               # HTML template
│   ├── package.json             # Node dependencies
│   └── tailwind.config.js       # Tailwind CSS config
└── README.md                    # This file
```

## 🔧 Configuration

### Backend Configuration

Edit `backend/app/config.py` to adjust quality thresholds:

```python
# Standard quality thresholds
MIN_WIDTH = 1000              # Minimum image width
MIN_HEIGHT = 1000             # Minimum image height
BLUR_THRESHOLD = 100.0        # Blur detection threshold
MIN_SHARPNESS = 50.0          # Sharpness threshold
MIN_BRIGHTNESS = 60           # Minimum brightness
MAX_BRIGHTNESS = 200          # Maximum brightness
MIN_BACKGROUND_SCORE = 0.7    # Background quality threshold

# CLIP configuration (new)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP model
CLIP_SIMILARITY_THRESHOLD = 0.25                   # Mismatch threshold
CLIP_MODEL_PATH = None                             # Fine-tuned model path
```

### Environment Variables (Optional)

Create `backend/.env` for CLIP configuration:

```bash
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
CLIP_SIMILARITY_THRESHOLD=0.25
CLIP_MODEL_PATH=./clip_checkpoints/best_model
CLIP_BATCH_SIZE=8
```

### Frontend Configuration

Edit `frontend/src/api/client.js` to change the API endpoint:

```javascript
const API = axios.create({
  baseURL: "http://localhost:8000"  // Backend URL
});
```

## 🧪 Example Requests

### Using cURL

```bash
# Upload with description
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/product.jpg" \
  -F "description=Red leather handbag with gold hardware"

# Get all results
curl http://localhost:8000/results

# Get specific result
curl http://localhost:8000/results/1
```

### Using Python

```python
import requests

# Upload image with description
url = "http://localhost:8000/upload"
files = {"file": open("product.jpg", "rb")}
data = {"description": "Blue cotton t-shirt"}
response = requests.post(url, files=files, data=data)
print(response.json())

# Get results
results = requests.get("http://localhost:8000/results")
print(results.json())
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: SQL toolkit and ORM
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computing
- **scikit-image**: Image processing algorithms

### Frontend
- **React 18**: UI library
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client
- **React Router**: Client-side routing

## 🐛 Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Use a different port
uvicorn app.main:app --reload --port 8001
```

**Database issues:**
```bash
# Remove the database file and restart
rm image_quality.db
uvicorn app.main:app --reload
```

**OpenCV import errors:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-opencv libgl1

# Or reinstall opencv-python
pip uninstall opencv-python
pip install opencv-python
```

### Frontend Issues

**Port 5173 already in use:**
```bash
# Kill the process
lsof -ti:5173 | xargs kill -9

# Or use different port
npm run dev -- --port 3000
```

**CORS errors:**
- Ensure backend is running at http://localhost:8000
- Check CORS settings in `backend/app/main.py`

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the GitHub repository.