from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .routes.upload import router as upload_router
from .routes.analyze import router as analyze_router
from .routes.results import router as results_router
from .routes.clip import router as clip_router

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Image Quality Analysis",
    description="Professional ecommerce product image quality analyzer with comprehensive metrics and CLIP-based mismatch detection",
    version="1.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router, tags=["Upload"])
app.include_router(analyze_router, tags=["Analysis"])
app.include_router(results_router, tags=["Results"])
app.include_router(clip_router, tags=["CLIP"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "service": "AI Image Quality Analysis",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok", "service": "running"}