"""
Autoencoder Neural Network for Anomaly Detection
Uses TensorFlow/Keras for reconstruction-based anomaly detection
"""
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from ..config import settings


class AutoencoderDetector:
    """Autoencoder-based anomaly detector using reconstruction error"""
    
    def __init__(
        self,
        input_dim: int = 7,
        latent_dim: int = None,
        epochs: int = None,
        batch_size: int = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim or settings.AUTOENCODER_LATENT_DIM
        self.epochs = epochs or settings.AUTOENCODER_EPOCHS
        self.batch_size = batch_size or settings.AUTOENCODER_BATCH_SIZE
        
        self.model: Optional[Model] = None
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self._is_trained = False
        self._threshold = 0.5
        self._model_path = settings.MODELS_DIR / "autoencoder.keras"
        
        self._build_model()
    
    def _build_model(self):
        """Build the autoencoder architecture"""
        # Encoder
        encoder_input = keras.Input(shape=(self.input_dim,), name='encoder_input')
        x = layers.Dense(5, activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        self.encoder = Model(encoder_input, latent, name='encoder')
        
        # Decoder
        decoder_input = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(5, activation='relu')(decoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        decoder_output = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        
        # Full autoencoder
        autoencoder_input = keras.Input(shape=(self.input_dim,), name='autoencoder_input')
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        
        self.model = Model(autoencoder_input, decoded, name='autoencoder')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
    
    def fit(self, X: np.ndarray, epochs: int = None, verbose: int = 0) -> "AutoencoderDetector":
        """Train the autoencoder on normal data"""
        train_epochs = epochs or self.epochs
        
        # Train only on data (assuming mostly normal)
        self.model.fit(
            X, X,
            epochs=train_epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=verbose,
            validation_split=0.1
        )
        
        # Calculate threshold based on training reconstruction error
        predictions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        self._threshold = np.percentile(mse, 95)  # 95th percentile
        
        self._is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly score based on reconstruction error
        Returns: array of scores between 0 (normal) and 1 (anomaly)
        """
        if not self._is_trained or self.model is None:
            return np.full(len(X) if hasattr(X, '__len__') else 1, 0.5)
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get reconstruction
        reconstructed = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction error (MSE)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        
        # Normalize based on threshold
        scores = mse / (self._threshold * 2)
        scores = np.clip(scores, 0, 1)
        
        return scores
    
    def predict_single(self, features: List[float]) -> float:
        """Predict anomaly score for a single sample"""
        X = np.array(features).reshape(1, -1)
        return float(self.predict(X)[0])
    
    def get_latent(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation"""
        if self.encoder is None:
            return np.zeros((len(X), self.latent_dim))
        return self.encoder.predict(X, verbose=0)
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk"""
        if self.model is not None:
            save_path = path or self._model_path
            self.model.save(save_path)
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load model from disk"""
        load_path = path or self._model_path
        if load_path.exists():
            self.model = keras.models.load_model(load_path)
            self._is_trained = True
            return True
        return False
