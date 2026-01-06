"""
Configuration settings for the ML Backend
"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "saved_models"
    
    # ML Settings
    ISOLATION_FOREST_CONTAMINATION: float = 0.1
    ISOLATION_FOREST_N_ESTIMATORS: int = 100
    
    KMEANS_N_CLUSTERS: int = 5
    KMEANS_MAX_ITER: int = 300
    
    KNN_N_NEIGHBORS: int = 5
    
    AUTOENCODER_EPOCHS: int = 50
    AUTOENCODER_BATCH_SIZE: int = 32
    AUTOENCODER_LATENT_DIM: int = 3
    
    # Ensemble Weights
    DEFAULT_ISOLATION_FOREST_WEIGHT: float = 0.30
    DEFAULT_AUTOENCODER_WEIGHT: float = 0.25
    DEFAULT_KMEANS_WEIGHT: float = 0.20
    DEFAULT_KNN_WEIGHT: float = 0.25
    
    # RLHF
    RLHF_LEARNING_RATE: float = 0.05
    RLHF_MIN_FEEDBACK_FOR_ADJUSTMENT: int = 10
    
    # Auto-Response
    AUTO_RESPONSE_ENABLED: bool = True
    AUTO_RESPONSE_THREAT_THRESHOLD: float = 0.85
    AUTO_RESPONSE_BLOCK_DURATION: int = 60  # minutes
    
    # Auto-Training
    AUTO_TRAINING_ENABLED: bool = True
    AUTO_TRAINING_MIN_SAMPLES: int = 100
    AUTO_TRAINING_MAX_SAMPLES: int = 10000
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
