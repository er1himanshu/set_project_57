# AI-Powered Image Quality Analysis and Management System

A full-stack application for analyzing image quality with AI-powered metrics including resolution, blur detection, brightness, and contrast analysis.

## Features
- Upload product images through a modern web interface
- Automatic quality analysis with multiple metrics:
  - Resolution validation (minimum 1000x1000)
  - Blur detection using Laplacian variance
  - Brightness analysis
  - Contrast scoring
- Pass/Fail decision with detailed reasoning
- Interactive results dashboard
- RESTful API with CORS support

## Prerequisites
- Python 3.8+
- Node.js 16+
- pip
- npm

## Setup and Run

### Backend (FastAPI)
```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server (runs on http://localhost:8000)
uvicorn app.main:app --reload
```

The backend API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Frontend (React + Vite)
```bash
cd frontend

# Install dependencies
npm install

# Start development server (runs on http://localhost:5173)
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

- `POST /upload` - Upload an image for analysis
- `GET /results` - Get all analysis results
- `GET /results/{id}` - Get specific result details
- `GET /analyze/{image_id}` - Get analysis for a specific image

## Quality Criteria

Images are evaluated based on:
- **Resolution**: Minimum 1000x1000 pixels
- **Blur**: Laplacian variance threshold of 100.0
- **Brightness**: Range of 60-200

Images failing any criterion will be marked as FAIL with detailed reasons.

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── models.py         # Database models
│   │   ├── schemas.py        # Pydantic schemas
│   │   ├── config.py         # Configuration
│   │   ├── database.py       # Database setup
│   │   ├── routes/           # API endpoints
│   │   └── services/         # Business logic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/           # React pages
│   │   ├── components/      # React components
│   │   └── api/             # API client
│   └── package.json
└── README.md
```