"""
Metrics API Endpoints
"""
from fastapi import APIRouter, HTTPException
import numpy as np

from ..services import detection_service
from ..ml.metrics import get_all_model_metrics
from ..ml import TrainingDataGenerator

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("")
async def get_metrics():
    """Get performance metrics for all ML models"""
    try:
        # Generate test data for evaluation
        generator = TrainingDataGenerator()
        X_test, y_test, _ = generator.generate_feature_matrix(n_samples=200, anomaly_ratio=0.3)
        
        # Get detector
        detector = detection_service.get_detector()
        
        # Calculate metrics
        metrics = get_all_model_metrics(detector, X_test, y_test)
        
        return {
            'metrics': metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model}")
async def get_model_metrics(model: str):
    """Get metrics for a specific model"""
    try:
        # Generate test data
        generator = TrainingDataGenerator()
        X_test, y_test, _ = generator.generate_feature_matrix(n_samples=100, anomaly_ratio=0.3)
        
        # Get detector
        detector = detection_service.get_detector()
        models = detector.get_models()
        
        # Map model name
        model_key = {
            'isolation-forest': 'isolation_forest',
            'isolation_forest': 'isolation_forest',
            'autoencoder': 'autoencoder',
            'kmeans': 'kmeans',
            'k-means': 'kmeans',
            'knn': 'knn',
        }.get(model.lower())
        
        if not model_key or model_key not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
        
        # Calculate metrics for specific model
        from ..ml.metrics import evaluate_model
        metrics = evaluate_model(models[model_key], X_test, y_test)
        
        method_name = {
            'isolation_forest': 'Isolation Forest',
            'autoencoder': 'Autoencoder',
            'kmeans': 'K-Means Clustering',
            'knn': 'KNN'
        }.get(model_key, model)
        
        metrics['method'] = method_name
        
        return metrics
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights")
async def get_weights():
    """Get current ensemble weights"""
    return {
        'weights': detection_service.get_weights()
    }
