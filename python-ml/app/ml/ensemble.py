"""
Ensemble Detector
Combines all ML methods for robust anomaly detection with configurable weights
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .isolation_forest import IsolationForestDetector
from .autoencoder import AutoencoderDetector
from .kmeans import KMeansDetector
from .knn import KNNDetector
from .features import extract_features
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeights:
    """Weights for ensemble voting"""
    isolation_forest: float = 0.30
    autoencoder: float = 0.25
    kmeans: float = 0.20
    knn: float = 0.25
    
    def normalize(self) -> "EnsembleWeights":
        """Normalize weights to sum to 1"""
        total = self.isolation_forest + self.autoencoder + self.kmeans + self.knn
        if total > 0:
            return EnsembleWeights(
                isolation_forest=self.isolation_forest / total,
                autoencoder=self.autoencoder / total,
                kmeans=self.kmeans / total,
                knn=self.knn / total
            )
        return self
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'isolationForest': self.isolation_forest,
            'autoencoder': self.autoencoder,
            'kMeans': self.kmeans,
            'knn': self.knn
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EnsembleWeights":
        """Create from dictionary"""
        return cls(
            isolation_forest=data.get('isolationForest', data.get('isolation_forest', 0.30)),
            autoencoder=data.get('autoencoder', 0.25),
            kmeans=data.get('kMeans', data.get('kmeans', data.get('k_means', 0.20))),
            knn=data.get('knn', 0.25)
        )


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction"""
    score: float
    is_anomaly: bool
    scores: Dict[str, float]
    attack_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'score': self.score,
            'isAnomaly': self.is_anomaly,
            'scores': {
                'isolationForest': self.scores.get('isolation_forest', 0),
                'autoencoder': self.scores.get('autoencoder', 0),
                'kMeans': self.scores.get('kmeans', 0),
                'knn': self.scores.get('knn', 0)
            },
            'attackType': self.attack_type
        }


class EnsembleDetector:
    """
    Ensemble anomaly detector combining multiple ML methods
    """
    
    def __init__(self, weights: Optional[EnsembleWeights] = None):
        # Initialize individual detectors
        self.isolation_forest = IsolationForestDetector()
        self.autoencoder = AutoencoderDetector()
        self.kmeans = KMeansDetector()
        self.knn = KNNDetector()
        
        # Set weights
        self.weights = weights or EnsembleWeights(
            isolation_forest=settings.DEFAULT_ISOLATION_FOREST_WEIGHT,
            autoencoder=settings.DEFAULT_AUTOENCODER_WEIGHT,
            kmeans=settings.DEFAULT_KMEANS_WEIGHT,
            knn=settings.DEFAULT_KNN_WEIGHT
        )
        
        self.anomaly_threshold = 0.5
        self._is_initialized = False
    
    def fit(
        self, 
        X: np.ndarray, 
        labels: Optional[List[bool]] = None,
        attack_types: Optional[List[Optional[str]]] = None,
        verbose: int = 0
    ) -> "EnsembleDetector":
        """
        Train all models with the provided data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Optional labels for supervised methods
            attack_types: Optional attack type labels
            verbose: Verbosity level
        """
        logger.info(f"Training ensemble on {len(X)} samples")
        
        # Train unsupervised models
        self.isolation_forest.fit(X)
        self.autoencoder.fit(X, verbose=verbose)
        self.kmeans.fit(X)
        
        # Train KNN (needs labels)
        if labels is not None and len(labels) == len(X):
            self.knn.fit(X, labels, attack_types)
        else:
            # Generate pseudo-labels from unsupervised models
            pseudo_labels = self._generate_pseudo_labels(X)
            self.knn.fit(X, pseudo_labels, attack_types)
        
        self._is_initialized = True
        logger.info("Ensemble training complete")
        return self
    
    def _generate_pseudo_labels(self, X: np.ndarray) -> List[bool]:
        """Generate pseudo-labels from unsupervised models"""
        if_scores = self.isolation_forest.predict(X)
        ae_scores = self.autoencoder.predict(X)
        km_scores = self.kmeans.predict(X)
        
        # Average score
        avg_scores = (if_scores + ae_scores + km_scores) / 3
        
        return [score > 0.5 for score in avg_scores]
    
    def predict(self, features: List[float]) -> EnsemblePrediction:
        """
        Predict anomaly score for a single sample
        
        Args:
            features: Feature vector
            
        Returns:
            EnsemblePrediction with score, is_anomaly, individual scores, and attack type
        """
        X = np.array(features).reshape(1, -1)
        
        # Get individual scores
        if_score = self.isolation_forest.predict_single(features)
        ae_score = min(self.autoencoder.predict_single(features), 1.0)
        km_score = min(self.kmeans.predict_single(features), 1.0)
        knn_result = self.knn.predict_single(features)
        knn_score = knn_result['confidence'] if knn_result['is_anomaly'] else 1 - knn_result['confidence']
        
        # Weighted ensemble score
        ensemble_score = (
            self.weights.isolation_forest * if_score +
            self.weights.autoencoder * ae_score +
            self.weights.kmeans * km_score +
            self.weights.knn * knn_score
        )
        
        is_anomaly = ensemble_score > self.anomaly_threshold
        
        return EnsemblePrediction(
            score=ensemble_score,
            is_anomaly=is_anomaly,
            scores={
                'isolation_forest': if_score,
                'autoencoder': ae_score,
                'kmeans': km_score,
                'knn': knn_score
            },
            attack_type=knn_result['attack_type'] if is_anomaly else None
        )
    
    def predict_batch(self, X: np.ndarray) -> List[EnsemblePrediction]:
        """Predict for multiple samples"""
        return [self.predict(x.tolist()) for x in X]
    
    def predict_by_method(self, features: List[float], method: str) -> float:
        """Get prediction from a specific method"""
        method_lower = method.lower()
        
        if 'isolation' in method_lower:
            return self.isolation_forest.predict_single(features)
        elif 'autoencoder' in method_lower:
            return self.autoencoder.predict_single(features)
        elif 'means' in method_lower or 'kmeans' in method_lower:
            return self.kmeans.predict_single(features)
        elif 'knn' in method_lower:
            result = self.knn.predict_single(features)
            return result['confidence']
        else:
            return self.predict(features).score
    
    def update_weights(self, new_weights: Dict[str, float]) -> EnsembleWeights:
        """Update ensemble weights (used by RLHF)"""
        self.weights = EnsembleWeights.from_dict(new_weights).normalize()
        return self.weights
    
    def get_weights(self) -> EnsembleWeights:
        """Get current weights"""
        return self.weights
    
    def set_anomaly_threshold(self, threshold: float):
        """Set anomaly threshold"""
        self.anomaly_threshold = max(0, min(1, threshold))
    
    def add_knn_sample(
        self, 
        features: List[float], 
        is_anomaly: bool, 
        attack_type: Optional[str] = None
    ):
        """Add training data to KNN for online learning"""
        self.knn.add_training_point(features, is_anomaly, attack_type)
    
    def is_trained(self) -> bool:
        """Check if all models are trained"""
        return (
            self.isolation_forest.is_trained() and
            self.autoencoder.is_trained() and
            self.kmeans.is_trained() and
            self.knn.is_trained()
        )
    
    def save_all(self):
        """Save all models to disk"""
        self.isolation_forest.save()
        self.autoencoder.save()
        self.kmeans.save()
        self.knn.save()
        logger.info("All models saved")
    
    def load_all(self) -> bool:
        """Load all models from disk"""
        success = (
            self.isolation_forest.load() and
            self.autoencoder.load() and
            self.kmeans.load() and
            self.knn.load()
        )
        if success:
            self._is_initialized = True
            logger.info("All models loaded")
        return success
    
    def get_models(self) -> Dict:
        """Get individual model instances"""
        return {
            'isolation_forest': self.isolation_forest,
            'autoencoder': self.autoencoder,
            'kmeans': self.kmeans,
            'knn': self.knn
        }
