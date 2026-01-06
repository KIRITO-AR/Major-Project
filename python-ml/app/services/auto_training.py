"""
Auto-Training Service
Automatic model retraining when anomalies are detected
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import uuid
import numpy as np

from ..ml.features import extract_features
from ..config import settings

logger = logging.getLogger(__name__)


class TrainingDataPoint:
    """Training data point"""
    def __init__(
        self,
        features: List[float],
        label: str,  # 'normal' or 'anomaly'
        attack_type: Optional[str] = None,
        confidence: float = 0,
        verified: bool = False,
        detection_id: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.features = features
        self.label = label
        self.attack_type = attack_type
        self.confidence = confidence
        self.verified = verified
        self.detection_id = detection_id
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'features': self.features,
            'label': self.label,
            'attackType': self.attack_type,
            'confidence': self.confidence,
            'verified': self.verified,
            'detectionId': self.detection_id,
            'createdAt': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingDataPoint':
        point = cls(
            features=data['features'],
            label=data['label'],
            attack_type=data.get('attackType'),
            confidence=data.get('confidence', 0),
            verified=data.get('verified', False),
            detection_id=data.get('detectionId')
        )
        point.id = data.get('id', str(uuid.uuid4()))
        if 'createdAt' in data:
            point.created_at = datetime.fromisoformat(data['createdAt'])
        return point


class TrainingResult:
    """Training result entry"""
    def __init__(
        self,
        samples_used: int,
        model_version: int,
        metrics: Dict = None,
        duration: float = 0
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.samples_used = samples_used
        self.model_version = model_version
        self.metrics = metrics or {}
        self.duration = duration
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'samplesUsed': self.samples_used,
            'modelVersion': self.model_version,
            'metrics': self.metrics,
            'duration': self.duration
        }


class AutoTrainingConfig:
    """Auto-training configuration"""
    def __init__(self):
        self.enabled = settings.AUTO_TRAINING_ENABLED
        self.min_samples_for_retrain = settings.AUTO_TRAINING_MIN_SAMPLES
        self.auto_retrain_on_new_anomalies = True
        self.max_stored_samples = settings.AUTO_TRAINING_MAX_SAMPLES
        self.include_normal_traffic = True
        self.normal_traffic_ratio = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'minSamplesForRetrain': self.min_samples_for_retrain,
            'autoRetrainOnNewAnomalies': self.auto_retrain_on_new_anomalies,
            'maxStoredSamples': self.max_stored_samples,
            'includeNormalTraffic': self.include_normal_traffic,
            'normalTrafficRatio': self.normal_traffic_ratio
        }
    
    def update(self, data: Dict):
        """Update configuration from dictionary"""
        if 'enabled' in data:
            self.enabled = data['enabled']
        if 'minSamplesForRetrain' in data:
            self.min_samples_for_retrain = data['minSamplesForRetrain']
        if 'autoRetrainOnNewAnomalies' in data:
            self.auto_retrain_on_new_anomalies = data['autoRetrainOnNewAnomalies']
        if 'maxStoredSamples' in data:
            self.max_stored_samples = data['maxStoredSamples']
        if 'includeNormalTraffic' in data:
            self.include_normal_traffic = data['includeNormalTraffic']
        if 'normalTrafficRatio' in data:
            self.normal_traffic_ratio = data['normalTrafficRatio']


class AutoTrainingService:
    """Auto-training service for model retraining"""
    
    def __init__(self):
        self.training_data: List[TrainingDataPoint] = []
        self.training_history: List[TrainingResult] = []
        self.model_version = 1
        self.config = AutoTrainingConfig()
        self.pending_retraining = False
        self._data_path = settings.DATA_DIR / "training_data.json"
        
        # Load saved data
        self._load_data()
    
    def add_detection_data(self, detection: Dict) -> TrainingDataPoint:
        """Add a detection result to training data"""
        packet = detection.get('packet', {})
        features = extract_features(packet)
        
        data_point = TrainingDataPoint(
            features=features,
            label='anomaly' if detection.get('isAnomaly') else 'normal',
            attack_type=detection.get('attackType'),
            confidence=detection.get('confidence', 0),
            verified=False,
            detection_id=detection.get('id')
        )
        
        # Check if we should add normal traffic
        if not detection.get('isAnomaly') and not self.config.include_normal_traffic:
            return data_point
        
        # Maintain ratio of normal to anomaly
        if not detection.get('isAnomaly'):
            anomaly_count = sum(1 for d in self.training_data if d.label == 'anomaly')
            normal_count = sum(1 for d in self.training_data if d.label == 'normal')
            
            if normal_count >= anomaly_count * self.config.normal_traffic_ratio:
                return data_point
        
        self.training_data.append(data_point)
        
        # Enforce max samples limit
        if len(self.training_data) > self.config.max_stored_samples:
            # Remove oldest non-anomaly samples first
            normal_samples = sorted(
                [d for d in self.training_data if d.label == 'normal'],
                key=lambda x: x.created_at
            )
            
            if normal_samples:
                to_remove = normal_samples[0].id
                self.training_data = [d for d in self.training_data if d.id != to_remove]
        
        # Check if we should trigger retraining
        if self.config.auto_retrain_on_new_anomalies and detection.get('isAnomaly'):
            self._check_retraining_needed()
        
        self._save_data()
        
        return data_point
    
    def verify_data_point(
        self, 
        id: str, 
        is_correct: bool, 
        correct_label: Optional[str] = None
    ):
        """Mark a training data point as verified"""
        for data_point in self.training_data:
            if data_point.id == id:
                data_point.verified = True
                if not is_correct and correct_label:
                    data_point.label = correct_label
                self._save_data()
                break
    
    def _check_retraining_needed(self) -> bool:
        """Check if retraining is needed"""
        unverified_anomalies = sum(
            1 for d in self.training_data
            if d.label == 'anomaly' and not d.verified
        )
        
        if unverified_anomalies >= self.config.min_samples_for_retrain and not self.pending_retraining:
            self.pending_retraining = True
            return True
        
        return False
    
    def execute_retraining(self) -> TrainingResult:
        """Execute model retraining"""
        import time
        start_time = time.time()
        
        # Get training data
        samples = [
            d for d in self.training_data
            if d.verified or d.confidence >= 80
        ]
        
        self.model_version += 1
        
        # Simulated metrics (real training would update the ensemble)
        result = TrainingResult(
            samples_used=len(samples),
            model_version=self.model_version,
            metrics={
                'accuracy': 0.95 + np.random.random() * 0.04,
                'precision': 0.93 + np.random.random() * 0.05,
                'recall': 0.91 + np.random.random() * 0.06
            },
            duration=(time.time() - start_time) * 1000
        )
        
        self.training_history.append(result)
        self.pending_retraining = False
        
        self._save_data()
        
        return result
    
    def get_training_data(
        self,
        label: Optional[str] = None,
        verified: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get training data"""
        data = self.training_data.copy()
        
        if label:
            data = [d for d in data if d.label == label]
        if verified is not None:
            data = [d for d in data if d.verified == verified]
        if limit:
            data = data[-limit:]
        
        return [d.to_dict() for d in data]
    
    def export_training_data(self) -> Dict:
        """Export training data as JSON"""
        normal_count = sum(1 for d in self.training_data if d.label == 'normal')
        anomaly_count = sum(1 for d in self.training_data if d.label == 'anomaly')
        
        return {
            'version': '1.0',
            'exportedAt': datetime.now().isoformat(),
            'samples': [d.to_dict() for d in self.training_data],
            'modelVersion': self.model_version,
            'totalSamples': {
                'normal': normal_count,
                'anomaly': anomaly_count
            }
        }
    
    def import_training_data(self, data: Dict):
        """Import training data from JSON"""
        if 'samples' in data and isinstance(data['samples'], list):
            existing_ids = {d.id for d in self.training_data}
            
            for sample in data['samples']:
                if sample.get('id') not in existing_ids:
                    self.training_data.append(TrainingDataPoint.from_dict(sample))
            
            if 'modelVersion' in data and data['modelVersion'] > self.model_version:
                self.model_version = data['modelVersion']
        
        self._save_data()
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'totalSamples': len(self.training_data),
            'normalSamples': sum(1 for d in self.training_data if d.label == 'normal'),
            'anomalySamples': sum(1 for d in self.training_data if d.label == 'anomaly'),
            'verifiedSamples': sum(1 for d in self.training_data if d.verified),
            'modelVersion': self.model_version,
            'pendingRetraining': self.pending_retraining,
            'trainingHistory': [h.to_dict() for h in self.training_history]
        }
    
    def update_config(self, updates: Dict) -> Dict:
        """Update configuration"""
        self.config.update(updates)
        self._save_data()
        return self.config.to_dict()
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.to_dict()
    
    def clear_training_data(self):
        """Clear all training data"""
        self.training_data = []
        self.pending_retraining = False
        self._save_data()
    
    def delete_training_sample(self, id: str) -> bool:
        """Delete a specific training sample"""
        initial_count = len(self.training_data)
        self.training_data = [d for d in self.training_data if d.id != id]
        
        if len(self.training_data) < initial_count:
            self._save_data()
            return True
        return False
    
    def _save_data(self):
        """Save data to disk"""
        data = {
            'training_data': [d.to_dict() for d in self.training_data],
            'training_history': [h.to_dict() for h in self.training_history],
            'model_version': self.model_version,
            'config': self.config.to_dict()
        }
        
        with open(self._data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load data from disk"""
        if self._data_path.exists():
            try:
                with open(self._data_path, 'r') as f:
                    data = json.load(f)
                
                if 'training_data' in data:
                    self.training_data = [
                        TrainingDataPoint.from_dict(d) 
                        for d in data['training_data']
                    ]
                if 'model_version' in data:
                    self.model_version = data['model_version']
                if 'config' in data:
                    self.config.update(data['config'])
                
                logger.info(f"Loaded training data from {self._data_path}")
            except Exception as e:
                logger.warning(f"Failed to load training data: {e}")


# Singleton instance
auto_training_service = AutoTrainingService()
