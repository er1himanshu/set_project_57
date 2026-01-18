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
- **CLIP-Powered Mismatch Detection**: AI-based image-text consistency validation using OpenAI's CLIP model

### AI-Powered Features
- **CLIP Integration**: Advanced neural network for understanding image-text relationships
- **Semantic Matching**: Goes beyond color matching to understand product descriptions
- **Fine-tuning Support**: Train custom models on your specific product domain
- **Configurable Thresholds**: Adjust sensitivity based on your needs

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

# Optional: Configure CLIP settings (see backend/.env.example)
cp .env.example .env
# Edit .env to customize CLIP model, threshold, device, etc.

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

### Upload Image
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

### CLIP Mismatch Detection
```http
POST /clip/check-mismatch
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - description: product description text (required)
  - threshold: similarity threshold (optional, default: 0.25)

Response:
{
  "is_match": true,
  "similarity_score": 0.87,
  "decision": "Match (score: 0.870 >= 0.250)",
  "threshold_used": 0.25
}
```

For detailed API documentation, see [docs/api-spec.md](docs/api-spec.md)

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
| **Description** | Consistent | CLIP-based semantic matching |
| **CLIP Similarity** | ≥ 0.6 | AI-powered image-text similarity score |

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
  "clip_similarity_score": 0.87,
  "clip_mismatch": false
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
│   │   │   ├── results.py       # Results retrieval endpoints
│   │   │   └── clip.py          # CLIP mismatch detection endpoints
│   │   └── services/
│   │       ├── image_quality.py # Image analysis logic
│   │       ├── storage.py       # File storage management
│   │       └── clip_service.py  # CLIP-based mismatch detection
│   ├── training/                # CLIP fine-tuning pipeline
│   │   ├── dataset_loader.py    # Dataset loader for training
│   │   ├── train_clip.py        # Training script
│   │   ├── inference_clip.py    # Standalone inference script
│   │   ├── README.md            # Training documentation
│   │   └── sample_data/         # Sample dataset examples
│   ├── requirements.txt         # Python dependencies
│   ├── .env.example             # Environment configuration example
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
├── docs/
│   ├── api-spec.md              # API documentation
│   ├── architecture.md          # Architecture overview
│   └── clip-usage-guide.md      # CLIP usage guide
└── README.md                    # This file
```

## 🔧 Configuration

### Backend Configuration

Edit `backend/app/config.py` or use environment variables (see `backend/.env.example`):

```python
# Image Quality Thresholds
MIN_WIDTH = 1000              # Minimum image width
MIN_HEIGHT = 1000             # Minimum image height
BLUR_THRESHOLD = 100.0        # Blur detection threshold
MIN_SHARPNESS = 50.0          # Sharpness threshold
MIN_BRIGHTNESS = 60           # Minimum brightness
MAX_BRIGHTNESS = 200          # Maximum brightness
MIN_BACKGROUND_SCORE = 0.7    # Background quality threshold

# CLIP Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP model
CLIP_SIMILARITY_THRESHOLD = 0.6  # Mismatch detection threshold
CLIP_DEVICE = "cpu"  # Use "cuda" for GPU acceleration
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

# Check image-text mismatch using CLIP
url = "http://localhost:8000/clip/check-mismatch"
files = {"file": open("product.jpg", "rb")}
data = {"description": "Red leather handbag", "threshold": 0.25}
response = requests.post(url, files=files, data=data)
print(response.json())
# Output: {"is_match": true, "similarity_score": 0.87, ...}
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: SQL toolkit and ORM
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computing
- **scikit-image**: Image processing algorithms
- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: CLIP model implementation
- **Pillow**: Image processing for CLIP

### Frontend
- **React 18**: UI library
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client
- **React Router**: Client-side routing

## 🤖 CLIP Fine-Tuning (Advanced)

For domain-specific accuracy, you can fine-tune CLIP on your product dataset:

### 1. Prepare Training Data

Create a CSV file with labeled image-text pairs:

```csv
image_path,text,label
products/handbag1.jpg,Red leather handbag,1
products/handbag2.jpg,Blue cotton t-shirt,0
```

- Label `1` = Match (consistent)
- Label `0` = Mismatch (inconsistent)

### 2. Train Model

```bash
cd backend/training

python train_clip.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --image_base_path data/ \
  --output_dir ./fine_tuned_clip \
  --epochs 10 \
  --batch_size 32 \
  --device cuda  # Use 'cpu' if no GPU
```

### 3. Use Fine-Tuned Model

```bash
export CLIP_FINE_TUNED_MODEL_PATH=/path/to/fine_tuned_clip/best_model
uvicorn app.main:app --reload
```

For detailed training instructions, see [backend/training/README.md](backend/training/README.md) and [docs/clip-usage-guide.md](docs/clip-usage-guide.md).

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

### CLIP Issues

**CLIP model loading slow:**
- First load downloads model (~350MB), subsequent loads are fast
- Models are cached in `backend/clip_models/`

**Out of memory (CUDA):**
```bash
# Use CPU instead
export CLIP_DEVICE=cpu
```

**Slow inference:**
- Use GPU for faster inference: `export CLIP_DEVICE=cuda`
- First request is slower (model loading), subsequent requests are fast

For more troubleshooting, see [docs/clip-usage-guide.md](docs/clip-usage-guide.md).

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the GitHub repository.