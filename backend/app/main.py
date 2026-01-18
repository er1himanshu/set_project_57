from fastapi import FastAPI
from .database import Base, engine
from .routes.upload import router as upload_router
from .routes.analyze import router as analyze_router
from .routes.results import router as results_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Image Quality Analysis")

app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(results_router)