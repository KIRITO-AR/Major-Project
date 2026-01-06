"""
K-Means Clustering for Anomaly Detection
Uses scikit-learn's KMeans for cluster-based anomaly detection
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional, List

from ..config import settings


class KMeansDetector:
    """K-Means clustering based anomaly detector"""
    
    def __init__(
        self,
        n_clusters: int = None,
        max_iter: int = None,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters or settings.KMEANS_N_CLUSTERS
        self.max_iter = max_iter or settings.KMEANS_MAX_ITER
        self.random_state = random_state
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10
        )
        self.scaler = StandardScaler()
        self._is_trained = False
        self._threshold = 1.0  # Distance threshold for anomaly
        self._model_path = settings.MODELS_DIR / "kmeans.joblib"
    
    def fit(self, X: np.ndarray) -> "KMeansDetector":
        """Train the K-Means model"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.model.fit(X_scaled)
        
        # Calculate threshold based on training distances
        distances = self._get_distances(X_scaled)
        self._threshold = np.percentile(distances, 95)
        
        self._is_trained = True
        return self
    
    def _get_distances(self, X: np.ndarray) -> np.ndarray:
        """Get distances to nearest cluster centroid"""
        # Get cluster assignments
        labels = self.model.predict(X)
        centroids = self.model.cluster_centers_
        
        # Calculate distances to assigned centroids
        distances = np.array([
            np.linalg.norm(X[i] - centroids[labels[i]])
            for i in range(len(X))
        ])
        
        return distances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly score based on distance from cluster centroids
        Returns: array of scores between 0 (normal) and 1 (anomaly)
        """
        if not self._is_trained:
            return np.full(len(X) if hasattr(X, '__len__') else 1, 0.5)
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get distances
        distances = self._get_distances(X_scaled)
        
        # Normalize based on threshold
        scores = distances / (self._threshold * 2)
        scores = np.clip(scores, 0, 1)
        
        return scores
    
    def predict_single(self, features: List[float]) -> float:
        """Predict anomaly score for a single sample"""
        X = np.array(features).reshape(1, -1)
        return float(self.predict(X)[0])
    
    def get_cluster(self, X: np.ndarray) -> np.ndarray:
        """Get cluster assignments"""
        if not self._is_trained:
            return np.zeros(len(X), dtype=int)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def save(self, path: Optional[Path] = None):
        """Save model and scaler to disk"""
        save_path = path or self._model_path
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self._threshold
        }, save_path)
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load model from disk"""
        load_path = path or self._model_path
        if load_path.exists():
            data = joblib.load(load_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self._threshold = data['threshold']
            self._is_trained = True
            return True
        return False
