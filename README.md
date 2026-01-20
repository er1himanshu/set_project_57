# AI-Powered Ecommerce Product Listing Evaluator

A full-stack application for analyzing product images with comprehensive AI-powered quality metrics specifically designed for ecommerce marketplaces. Ensure your product images meet professional standards with instant feedback and actionable suggestions.

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
- **🆕 Image-Text Mismatch Detection**: Uses pretrained CLIP (Contrastive Language-Image Pre-training) model to detect mismatches between product images and descriptions with AI-powered similarity scoring
- **🆕 CLIP Explainability**: Visual attention rollout heatmaps showing which image regions most influenced the CLIP similarity score, helping understand AI decision-making

### User Experience
- **Modern UI**: Beautiful gradient-based design with card layouts
- **Image Preview**: See your image before upload
- **Real-time Results**: Instant analysis with detailed quality checklist
- **Improvement Suggestions**: Actionable tips to enhance image quality
- **Results Dashboard**: Track all analyses with statistics and filters
- **🆕 AI Explainability Visualization**: Interactive heatmap overlays showing attention patterns from CLIP model

## 🚀 Prerequisites

- **Python** 3.8 or higher
- **Node.js** 16 or higher
- **pip** (Python package manager)
- **npm** (Node package manager)

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

# Install dependencies (this may take several minutes for CLIP model dependencies)
pip install -r requirements.txt

# Start the backend server
uvicorn app.main:app --reload
```

**Note**: The first time you upload an image with a description, the CLIP model (~350MB) will be automatically downloaded. This is a one-time download.

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

### Upload Image
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - description: product description text (optional, enables mismatch detection)

Response:
{
  "message": "Uploaded & analyzed",
  "result_id": 1,
  "passed": true
}
```

**Note**: When a description is provided, the system automatically runs CLIP-based mismatch detection alongside quality checks.

### Check Image-Text Mismatch
```http
POST /check-mismatch
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - description: product description text (required, min 10 characters)
  - threshold: similarity threshold 0-1 (optional, default: 0.25)

Response:
{
  "filename": "product_abc123.jpg",
  "description": "Red leather handbag",
  "has_mismatch": false,
  "similarity_score": 0.85,
  "threshold": 0.25,
  "message": "Match confirmed (score: 0.85)",
  "recommendation": "Image and description match well."
}
```

**Use this endpoint** when you specifically want to test mismatch detection without storing results in the database.

### 🆕 Explain CLIP Similarity (Explainability)
```http
POST /explain
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - description: product description text (required, min 10 characters)
  - threshold: similarity threshold 0-1 (optional, default: 0.25)

Response:
{
  "filename": "product_abc123.jpg",
  "description": "Red leather handbag with gold hardware",
  "similarity_score": 0.85,
  "has_mismatch": false,
  "threshold": 0.25,
  "message": "Match confirmed (score: 0.85)",
  "heatmap_base64": "<base64-encoded PNG image>",
  "explanation": "Heatmap shows which image regions most influenced the similarity score..."
}
```

**This endpoint** generates a visual explanation using CLIP attention rollout, showing which parts of the image most influenced the similarity score between the image and description. The heatmap overlay uses warmer colors (red/yellow) to indicate regions of higher attention, and cooler colors (blue) for lower attention. The returned `heatmap_base64` is a base64-encoded PNG image that can be displayed directly in the browser.

**Fallback Behavior**: If the CLIP model does not provide attention tensors (e.g., due to model architecture limitations or incomplete attention outputs), the endpoint automatically uses a simplified gradient-based visualization. The similarity score remains accurate and is always computed. The `explanation` field in the response will indicate when a fallback visualization is used.

**Requirements**: This feature requires the CLIP model to be available locally. If the model is not accessible, the endpoint returns a 503 error.

### Get All Results
```http
GET /results

Response: Array of ImageResultSchema objects
```

### Get Specific Result
```http
GET /results/{result_id}

Response: ImageResultSchema object with all quality metrics
```

### Get Analysis by Image ID
```http
GET /analyze/{image_id}

Response: ImageResultSchema object
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
| **🆕 Image-Text Match** | ≥ 0.25 | CLIP similarity score (0-1) |

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
  "improvement_suggestions": "Image meets quality standards",
  "has_mismatch": false,
  "similarity_score": 0.85,
  "mismatch_message": "Match confirmed (score: 0.85)"
}
```

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
│   │   │   ├── mismatch.py      # 🆕 Mismatch detection endpoint
│   │   │   ├── explain.py       # 🆕 CLIP explainability endpoint
│   │   │   └── results.py       # Results retrieval endpoints
│   │   └── services/
│   │       ├── image_quality.py # Image analysis logic
│   │       ├── mismatch_detector.py # 🆕 CLIP-based mismatch detection
│   │       ├── explainability.py # 🆕 CLIP attention rollout & visualization
│   │       └── storage.py       # File storage management
│   ├── requirements.txt         # Python dependencies
│   └── uploads/                 # Uploaded images (auto-created)
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
MIN_WIDTH = 1000              # Minimum image width
MIN_HEIGHT = 1000             # Minimum image height
BLUR_THRESHOLD = 100.0        # Blur detection threshold
MIN_SHARPNESS = 50.0          # Sharpness threshold
MIN_BRIGHTNESS = 60           # Minimum brightness
MAX_BRIGHTNESS = 200          # Maximum brightness
MIN_BACKGROUND_SCORE = 0.7    # Background quality threshold

# 🆕 Image-Text Mismatch Detection
MISMATCH_THRESHOLD = 0.25     # Similarity threshold (0-1, lower = stricter)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP model to use
```

**Note**: The image-text mismatch detection feature requires the CLIP model to be available locally. If the model is not cached and cannot be downloaded (e.g., in offline/sandboxed environments), the feature will be gracefully skipped and uploads will continue successfully without mismatch detection. The image quality analysis will still be performed normally.

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
# Upload with description (includes automatic mismatch detection)
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/product.jpg" \
  -F "description=Red leather handbag with gold hardware"

# 🆕 Check mismatch detection specifically
curl -X POST http://localhost:8000/check-mismatch \
  -F "file=@/path/to/product.jpg" \
  -F "description=Red leather handbag with gold hardware"

# 🆕 Check mismatch with custom threshold
curl -X POST http://localhost:8000/check-mismatch \
  -F "file=@/path/to/product.jpg" \
  -F "description=Red leather handbag" \
  -F "threshold=0.30"

# 🆕 Generate CLIP explanation with attention heatmap
curl -X POST http://localhost:8000/explain \
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

# Upload image with description (includes automatic mismatch detection)
url = "http://localhost:8000/upload"
files = {"file": open("product.jpg", "rb")}
data = {"description": "Blue cotton t-shirt"}
response = requests.post(url, files=files, data=data)
print(response.json())

# 🆕 Check mismatch detection specifically
url = "http://localhost:8000/check-mismatch"
files = {"file": open("product.jpg", "rb")}
data = {"description": "Blue cotton t-shirt"}
response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Has Mismatch: {result['has_mismatch']}")
print(f"Similarity Score: {result['similarity_score']}")
print(f"Message: {result['message']}")

# 🆕 Generate CLIP explanation with attention heatmap
url = "http://localhost:8000/explain"
files = {"file": open("product.jpg", "rb")}
data = {"description": "Blue cotton t-shirt"}
response = requests.post(url, files=files, data=data)
result = response.json()
print(f"Similarity Score: {result['similarity_score']}")
print(f"Has Mismatch: {result['has_mismatch']}")
print(f"Message: {result['message']}")
# Heatmap is available as base64: result['heatmap_base64']
# To display: f"data:image/png;base64,{result['heatmap_base64']}"

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
- **🆕 Transformers**: Hugging Face library for pretrained models
- **🆕 PyTorch**: Deep learning framework for CLIP model
- **🆕 Pillow**: Python Imaging Library for image handling

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

**🆕 CLIP model download issues:**
```bash
# The CLIP model (~350MB) downloads automatically on first use
# If download fails, check your internet connection and try again

# To pre-download the model:
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
```

**🆕 PyTorch installation issues:**
```bash
# If torch installation fails, visit https://pytorch.org/get-started/locally/
# for platform-specific installation instructions

# For CPU-only installation:
pip install torch --index-url https://download.pytorch.org/whl/cpu
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