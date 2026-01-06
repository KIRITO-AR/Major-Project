"""
API Response Models
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from .detection import DetectionResult, EnsembleWeights, AttackType


class DetectionSummary(BaseModel):
    """Summary of detection results"""
    total: int
    anomalies: int
    normal: int
    critical: int
    high: int
    medium: int
    low: int
    blocked: int = 0


class DetectResponse(BaseModel):
    """Response for detect endpoint"""
    results: List[Dict[str, Any]]
    summary: DetectionSummary
    weights: EnsembleWeights


class RLHFFeedbackRequest(BaseModel):
    """RLHF feedback submission"""
    detection_id: str = Field(..., alias="detectionId")
    is_correct: bool = Field(..., alias="isCorrect")
    correct_label: Optional[str] = Field(None, alias="correctLabel")
    correct_attack_type: Optional[AttackType] = Field(None, alias="correctAttackType")
    model_method: Optional[str] = Field(None, alias="modelMethod")
    notes: Optional[str] = None
    
    class Config:
        populate_by_name = True


class RLHFMetrics(BaseModel):
    """RLHF metrics response"""
    total_feedback: int = Field(..., alias="totalFeedback")
    correct_predictions: int = Field(..., alias="correctPredictions")
    incorrect_predictions: int = Field(..., alias="incorrectPredictions")
    accuracy_rate: float = Field(..., alias="accuracyRate")
    weight_adjustments: int = Field(..., alias="weightAdjustments")
    last_update: Optional[datetime] = Field(None, alias="lastUpdate")
    
    class Config:
        populate_by_name = True


class BlockedIP(BaseModel):
    """Blocked IP model"""
    id: str
    ip_address: str = Field(..., alias="ipAddress")
    reason: str
    attack_type: Optional[AttackType] = Field(None, alias="attackType")
    confidence: float
    blocked_at: datetime = Field(..., alias="blockedAt")
    expires_at: Optional[datetime] = Field(None, alias="expiresAt")
    auto_blocked: bool = Field(..., alias="autoBlocked")
    
    class Config:
        populate_by_name = True


class AutoResponseConfig(BaseModel):
    """Auto-response configuration"""
    enabled: bool = True
    threat_threshold: float = Field(0.85, alias="threatThreshold")
    auto_block_duration: int = Field(60, alias="autoBlockDuration")
    block_on_critical: bool = Field(True, alias="blockOnCritical")
    block_on_high: bool = Field(True, alias="blockOnHigh")
    block_on_medium: bool = Field(False, alias="blockOnMedium")
    notify_on_block: bool = Field(True, alias="notifyOnBlock")
    whitelisted_ips: List[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"], alias="whitelistedIPs")
    
    class Config:
        populate_by_name = True


class AutoResponseStats(BaseModel):
    """Auto-response statistics"""
    total_blocked: int = Field(..., alias="totalBlocked")
    auto_blocked: int = Field(..., alias="autoBlocked")
    manual_blocked: int = Field(..., alias="manualBlocked")
    total_events: int = Field(..., alias="totalEvents")
    
    class Config:
        populate_by_name = True


class TrainingDataPoint(BaseModel):
    """Training data point"""
    id: str
    features: List[float]
    label: str
    attack_type: Optional[str] = Field(None, alias="attackType")
    confidence: float
    verified: bool = False
    created_at: datetime = Field(..., alias="createdAt")
    detection_id: Optional[str] = Field(None, alias="detectionId")
    
    class Config:
        populate_by_name = True


class TrainingStats(BaseModel):
    """Training statistics"""
    total_samples: int = Field(..., alias="totalSamples")
    normal_samples: int = Field(..., alias="normalSamples")
    anomaly_samples: int = Field(..., alias="anomalySamples")
    verified_samples: int = Field(..., alias="verifiedSamples")
    model_version: int = Field(..., alias="modelVersion")
    pending_retraining: bool = Field(..., alias="pendingRetraining")
    
    class Config:
        populate_by_name = True


class ModelMetrics(BaseModel):
    """ML model performance metrics"""
    method: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float = Field(..., alias="f1Score")
    false_positive_rate: float = Field(..., alias="falsePositiveRate")
    detection_time: float = Field(..., alias="detectionTime")
    
    class Config:
        populate_by_name = True
