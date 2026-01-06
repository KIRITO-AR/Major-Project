"""
Isolation Forest Anomaly Detection
Uses scikit-learn's IsolationForest for real anomaly detection
"""
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
from typing import Optional, List

from ..config import settings


class IsolationForestDetector:
    """Isolation Forest based anomaly detector"""
    
    def __init__(
        self, 
        n_estimators: int = None, 
        contamination: float = None,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators or settings.ISOLATION_FOREST_N_ESTIMATORS
        self.contamination = contamination or settings.ISOLATION_FOREST_CONTAMINATION
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self._is_trained = False
        self._model_path = settings.MODELS_DIR / "isolation_forest.joblib"
    
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Train the Isolation Forest model"""
        self.model.fit(X)
        self._is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly score for samples
        Returns: array of scores between 0 (normal) and 1 (anomaly)
        """
        if not self._is_trained:
            # Return neutral scores if not trained
            return np.full(len(X) if hasattr(X, '__len__') else 1, 0.5)
        
        # Get anomaly scores (negative scores indicate anomalies in sklearn)
        scores = -self.model.score_samples(X)
        
        # Normalize to 0-1 range
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores)
        
        return normalized
    
    def predict_single(self, features: List[float]) -> float:
        """Predict anomaly score for a single sample"""
        X = np.array(features).reshape(1, -1)
        return float(self.predict(X)[0])
    
    def is_anomaly(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Determine if samples are anomalies"""
        scores = self.predict(X)
        return scores > threshold
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk"""
        save_path = path or self._model_path
        joblib.dump(self.model, save_path)
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load model from disk"""
        load_path = path or self._model_path
        if load_path.exists():
            self.model = joblib.load(load_path)
            self._is_trained = True
            return True
        return False
