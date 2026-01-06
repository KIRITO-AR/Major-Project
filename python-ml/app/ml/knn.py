"""
K-Nearest Neighbors Classifier for Anomaly Detection
Uses scikit-learn's KNN with attack type classification
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import Counter

from ..config import settings


class KNNDetector:
    """KNN-based anomaly detector with attack type classification"""
    
    def __init__(
        self,
        n_neighbors: int = None,
        weights: str = 'distance'
    ):
        self.n_neighbors = n_neighbors or settings.KNN_N_NEIGHBORS
        self.weights = weights
        
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self._is_trained = False
        self._model_path = settings.MODELS_DIR / "knn.joblib"
        
        # Store raw training data for online learning
        self._X_train: List[List[float]] = []
        self._y_train: List[bool] = []
        self._attack_types: List[Optional[str]] = []
    
    def fit(
        self, 
        X: np.ndarray, 
        labels: List[bool],
        attack_types: Optional[List[Optional[str]]] = None
    ) -> "KNNDetector":
        """Train the KNN model with labeled data"""
        # Store training data
        self._X_train = X.tolist() if isinstance(X, np.ndarray) else X
        self._y_train = list(labels)
        self._attack_types = attack_types or [None] * len(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert labels to integers
        y = np.array([1 if label else 0 for label in labels])
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        self._is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
        """
        Predict if samples are anomalies
        Returns: (is_anomaly, confidence, attack_types)
        """
        if not self._is_trained:
            n = len(X) if hasattr(X, '__len__') else 1
            return np.zeros(n, dtype=bool), np.full(n, 0.5), [None] * n
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get confidence (probability of predicted class)
        confidence = np.max(probabilities, axis=1)
        
        # Get attack types from neighbors
        attack_types = self._get_neighbor_attack_types(X_scaled)
        
        return predictions.astype(bool), confidence, attack_types
    
    def _get_neighbor_attack_types(self, X_scaled: np.ndarray) -> List[Optional[str]]:
        """Get most common attack type from neighbors"""
        if not self._X_train or not self._attack_types:
            return [None] * len(X_scaled)
        
        # Get neighbor indices
        distances, indices = self.model.kneighbors(X_scaled)
        
        attack_types = []
        for neighbor_indices in indices:
            # Get attack types of neighbors that are anomalies
            neighbor_attacks = [
                self._attack_types[i] 
                for i in neighbor_indices 
                if self._y_train[i] and self._attack_types[i]
            ]
            
            if neighbor_attacks:
                # Return most common attack type
                attack_types.append(Counter(neighbor_attacks).most_common(1)[0][0])
            else:
                attack_types.append(None)
        
        return attack_types
    
    def predict_single(self, features: List[float]) -> Dict:
        """Predict for a single sample"""
        X = np.array(features).reshape(1, -1)
        is_anomaly, confidence, attack_types = self.predict(X)
        
        return {
            'is_anomaly': bool(is_anomaly[0]),
            'confidence': float(confidence[0]),
            'attack_type': attack_types[0]
        }
    
    def get_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly score (probability of being anomaly)"""
        if not self._is_trained:
            return np.full(len(X) if hasattr(X, '__len__') else 1, 0.5)
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Return probability of class 1 (anomaly)
        if probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities[:, 0]
    
    def add_training_point(
        self, 
        features: List[float], 
        is_anomaly: bool, 
        attack_type: Optional[str] = None
    ):
        """Add a new training point (online learning)"""
        self._X_train.append(features)
        self._y_train.append(is_anomaly)
        self._attack_types.append(attack_type)
        
        # Retrain with updated data
        if len(self._X_train) >= self.n_neighbors:
            X = np.array(self._X_train)
            X_scaled = self.scaler.fit_transform(X)
            y = np.array([1 if label else 0 for label in self._y_train])
            self.model.fit(X_scaled, y)
            self._is_trained = True
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def get_training_size(self) -> int:
        """Get number of training samples"""
        return len(self._X_train)
    
    def save(self, path: Optional[Path] = None):
        """Save model and training data to disk"""
        save_path = path or self._model_path
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'X_train': self._X_train,
            'y_train': self._y_train,
            'attack_types': self._attack_types
        }, save_path)
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load model from disk"""
        load_path = path or self._model_path
        if load_path.exists():
            data = joblib.load(load_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self._X_train = data['X_train']
            self._y_train = data['y_train']
            self._attack_types = data['attack_types']
            self._is_trained = True
            return True
        return False
