"""
AI-Based IDS - Python ML Backend
FastAPI Application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import (
    detect_router,
    rlhf_router,
    auto_response_router,
    training_router,
    metrics_router
)
from .services import detection_service
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting AI-Based IDS ML Backend...")
    
    # Initialize detector on startup
    try:
        detection_service.initialize()
        logger.info("Detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI-Based IDS ML Backend...")


# Create FastAPI app
app = FastAPI(
    title="AI-Based IDS ML Backend",
    description="Python ML Backend for AI-Based Intrusion Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect_router)
app.include_router(rlhf_router)
app.include_router(auto_response_router)
app.include_router(training_router)
app.include_router(metrics_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI-Based IDS ML Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "detect": "/detect",
            "rlhf": "/rlhf",
            "auto-response": "/auto-response",
            "training": "/training",
            "metrics": "/metrics",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    detector = detection_service.detector
    
    return {
        "status": "healthy",
        "detector_initialized": detector is not None,
        "detector_trained": detector.is_trained() if detector else False
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
