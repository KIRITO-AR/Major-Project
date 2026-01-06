"""
Detection Result Models
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict
from datetime import datetime
import uuid

AttackType = Literal[
    'DoS', 'DDoS', 'Probe', 'R2L', 'U2R', 'Brute Force', 
    'Port Scan', 'SQL Injection', 'XSS', 'Malware', 
    'Botnet', 'Man-in-the-Middle', 'Unknown'
]

ThreatLevel = Literal['low', 'medium', 'high', 'critical']

DetectionMethod = Literal[
    'Isolation Forest', 'Autoencoder', 'K-Means Clustering', 'KNN', 'Ensemble'
]

AutoResponseAction = Literal['blocked', 'alerted', 'monitored', 'ignored']

class ModelScores(BaseModel):
    """Individual model scores"""
    isolation_forest: float = Field(..., alias="isolationForest")
    autoencoder: float
    k_means: float = Field(..., alias="kMeans")
    knn: float
    
    class Config:
        populate_by_name = True

class DetectionResult(BaseModel):
    """Detection result model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    packet: Dict  # NetworkPacket as dict for flexibility
    is_anomaly: bool = Field(..., alias="isAnomaly")
    threat_level: ThreatLevel = Field(..., alias="threatLevel")
    attack_type: Optional[AttackType] = Field(None, alias="attackType")
    confidence: float = Field(..., ge=0, le=100)
    detection_method: DetectionMethod = Field(..., alias="detectionMethod")
    description: str
    recommendations: list[str] = []
    model_scores: Optional[ModelScores] = Field(None, alias="modelScores")
    auto_response_action: Optional[AutoResponseAction] = Field(None, alias="autoResponseAction")
    
    class Config:
        populate_by_name = True

class EnsembleWeights(BaseModel):
    """Ensemble model weights"""
    isolation_forest: float = Field(0.30, alias="isolationForest")
    autoencoder: float = 0.25
    k_means: float = Field(0.20, alias="kMeans")
    knn: float = 0.25
    
    class Config:
        populate_by_name = True
    
    def normalize(self) -> "EnsembleWeights":
        """Normalize weights to sum to 1"""
        total = self.isolation_forest + self.autoencoder + self.k_means + self.knn
        if total > 0:
            return EnsembleWeights(
                isolation_forest=self.isolation_forest / total,
                autoencoder=self.autoencoder / total,
                k_means=self.k_means / total,
                knn=self.knn / total
            )
        return self
