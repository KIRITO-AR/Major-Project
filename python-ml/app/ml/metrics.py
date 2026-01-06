"""
Model Metrics Calculation
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import time


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    detection_times: List[float] = None
) -> Dict[str, float]:
    """
    Calculate performance metrics for a model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        detection_times: List of detection times in ms
        
    Returns:
        Dictionary of metrics
    """
    # Convert to binary if needed
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate false positive rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Average detection time
    avg_detection_time = np.mean(detection_times) if detection_times else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1Score': float(f1),
        'falsePositiveRate': float(fpr),
        'detectionTime': float(avg_detection_time)
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a specific model
    
    Args:
        model: Model with predict method
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    detection_times = []
    predictions = []
    
    for x in X_test:
        start_time = time.time()
        
        if hasattr(model, 'predict_single'):
            result = model.predict_single(x.tolist())
            if isinstance(result, dict):
                pred = result.get('is_anomaly', result.get('confidence', 0) > 0.5)
            else:
                pred = result > 0.5
        else:
            pred = model.predict(x.reshape(1, -1))[0] > 0.5
        
        end_time = time.time()
        
        predictions.append(pred)
        detection_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return calculate_metrics(y_test, np.array(predictions), detection_times)


def get_all_model_metrics(ensemble, X_test: np.ndarray, y_test: np.ndarray) -> List[Dict]:
    """
    Get metrics for all models in ensemble
    
    Args:
        ensemble: EnsembleDetector instance
        X_test: Test features
        y_test: Test labels
        
    Returns:
        List of metrics dictionaries for each model
    """
    models = ensemble.get_models()
    
    results = []
    
    for name, model in models.items():
        method_name = {
            'isolation_forest': 'Isolation Forest',
            'autoencoder': 'Autoencoder',
            'kmeans': 'K-Means Clustering',
            'knn': 'KNN'
        }.get(name, name)
        
        metrics = evaluate_model(model, X_test, y_test)
        metrics['method'] = method_name
        results.append(metrics)
    
    # Add ensemble metrics
    ensemble_predictions = []
    detection_times = []
    
    for x in X_test:
        start_time = time.time()
        result = ensemble.predict(x.tolist())
        end_time = time.time()
        
        ensemble_predictions.append(result.is_anomaly)
        detection_times.append((end_time - start_time) * 1000)
    
    ensemble_metrics = calculate_metrics(y_test, np.array(ensemble_predictions), detection_times)
    ensemble_metrics['method'] = 'Ensemble'
    results.append(ensemble_metrics)
    
    return results
