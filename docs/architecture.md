# Architecture

## Backend Stack

- **FastAPI**: Modern Python web framework with async support
- **OpenCV + Scikit-image**: Traditional image quality analysis
- **CLIP (Transformers)**: AI-powered image-text similarity for mismatch detection
- **PyTorch**: Deep learning framework for CLIP
- **SQLite**: Lightweight database for storing analysis results

## Frontend Stack

- **React + Tailwind**: Modern UI framework and styling

## CLIP Integration

### Pre-trained Model
- Uses OpenAI's CLIP model for zero-shot image-text matching
- Default: `openai/clip-vit-base-patch32`
- Supports GPU acceleration for faster inference

### Fine-tuning Pipeline
- Custom training scripts for domain-specific models
- Dataset loader for CSV-based training data
- Checkpoint saving and model evaluation
- Located in `backend/training/`

### Services
- `clip_service.py`: Core CLIP inference service
- Singleton pattern for efficient model loading
- Configurable threshold for match/mismatch decisions

## Data Flow

1. **Upload**: User uploads image + description
2. **Storage**: Image saved to uploads directory
3. **Analysis**: 
   - Traditional metrics computed (OpenCV)
   - CLIP similarity computed (if description provided)
4. **Database**: Results stored with CLIP scores
5. **Response**: Comprehensive analysis returned to frontend

## Configuration

Environment-based configuration via:
- `.env` file for local development
- Environment variables for production
- `config.py` for default values and constants