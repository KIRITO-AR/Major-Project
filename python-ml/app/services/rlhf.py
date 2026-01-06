"""
RLHF (Reinforcement Learning from Human Feedback) Service
Collects user feedback to improve model weights dynamically
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import uuid

from ..ml.ensemble import EnsembleWeights
from ..config import settings

logger = logging.getLogger(__name__)


class RLHFFeedback:
    """RLHF feedback entry"""
    def __init__(
        self,
        detection_id: str,
        is_correct: bool,
        correct_label: Optional[str] = None,
        attack_type: Optional[str] = None,
        model_method: Optional[str] = None,
        notes: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.detection_id = detection_id
        self.is_correct = is_correct
        self.correct_label = correct_label
        self.attack_type = attack_type
        self.model_method = model_method
        self.notes = notes
        self.feedback_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'detectionId': self.detection_id,
            'isCorrect': self.is_correct,
            'correctLabel': self.correct_label,
            'attackType': self.attack_type,
            'modelMethod': self.model_method,
            'notes': self.notes,
            'feedbackAt': self.feedback_at.isoformat()
        }


class WeightHistory:
    """Weight adjustment history entry"""
    def __init__(self, weights: EnsembleWeights, reason: str):
        self.timestamp = datetime.now()
        self.weights = weights
        self.reason = reason
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'weights': self.weights.to_dict(),
            'reason': self.reason
        }


class RLHFService:
    """RLHF feedback and weight adjustment service"""
    
    def __init__(self):
        self.feedback_history: List[RLHFFeedback] = []
        self.weight_history: List[WeightHistory] = []
        self.current_weights = EnsembleWeights(
            isolation_forest=settings.DEFAULT_ISOLATION_FOREST_WEIGHT,
            autoencoder=settings.DEFAULT_AUTOENCODER_WEIGHT,
            kmeans=settings.DEFAULT_KMEANS_WEIGHT,
            knn=settings.DEFAULT_KNN_WEIGHT
        )
        self.learning_rate = settings.RLHF_LEARNING_RATE
        self.min_adjustment_threshold = settings.RLHF_MIN_FEEDBACK_FOR_ADJUSTMENT
        self._data_path = settings.DATA_DIR / "rlhf_data.json"
        
        # Load saved data
        self._load_data()
    
    def add_feedback(
        self,
        detection_id: str,
        is_correct: bool,
        correct_label: Optional[str] = None,
        attack_type: Optional[str] = None,
        model_method: Optional[str] = None,
        notes: Optional[str] = None
    ) -> RLHFFeedback:
        """Record user feedback for a detection"""
        feedback = RLHFFeedback(
            detection_id=detection_id,
            is_correct=is_correct,
            correct_label=correct_label,
            attack_type=attack_type,
            model_method=model_method,
            notes=notes
        )
        
        self.feedback_history.append(feedback)
        
        # Check if we should adjust weights
        if len(self.feedback_history) % self.min_adjustment_threshold == 0:
            self.adjust_weights()
        
        # Save data
        self._save_data()
        
        return feedback
    
    def adjust_weights(self) -> EnsembleWeights:
        """Adjust model weights based on feedback"""
        recent_feedback = self.feedback_history[-100:]
        
        if len(recent_feedback) < self.min_adjustment_threshold:
            return self.current_weights
        
        # Calculate performance by method
        method_performance = {
            'Isolation Forest': {'correct': 0, 'total': 0},
            'Autoencoder': {'correct': 0, 'total': 0},
            'K-Means Clustering': {'correct': 0, 'total': 0},
            'KNN': {'correct': 0, 'total': 0}
        }
        
        for fb in recent_feedback:
            if fb.model_method and fb.model_method in method_performance:
                method_performance[fb.model_method]['total'] += 1
                if fb.is_correct:
                    method_performance[fb.model_method]['correct'] += 1
        
        # Calculate accuracies
        accuracies = {}
        total_accuracy = 0
        
        for method, perf in method_performance.items():
            accuracy = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
            accuracies[method] = accuracy
            total_accuracy += accuracy
        
        # Only adjust if we have meaningful data
        if total_accuracy > 0:
            old_weights = self.current_weights
            
            # Blend current weights with performance-based weights
            new_if = self._blend(
                old_weights.isolation_forest,
                accuracies['Isolation Forest'] / total_accuracy
            )
            new_ae = self._blend(
                old_weights.autoencoder,
                accuracies['Autoencoder'] / total_accuracy
            )
            new_km = self._blend(
                old_weights.kmeans,
                accuracies['K-Means Clustering'] / total_accuracy
            )
            new_knn = self._blend(
                old_weights.knn,
                accuracies['KNN'] / total_accuracy
            )
            
            self.current_weights = EnsembleWeights(
                isolation_forest=new_if,
                autoencoder=new_ae,
                kmeans=new_km,
                knn=new_knn
            ).normalize()
            
            # Record history
            self.weight_history.append(WeightHistory(
                weights=self.current_weights,
                reason=f"Based on {len(recent_feedback)} feedback entries"
            ))
            
            self._save_data()
        
        return self.current_weights
    
    def _blend(self, current: float, target: float) -> float:
        """Blend current and target values using learning rate"""
        return current * (1 - self.learning_rate) + target * self.learning_rate
    
    def get_weights(self) -> EnsembleWeights:
        """Get current weights"""
        return self.current_weights
    
    def set_learning_rate(self, rate: float):
        """Set learning rate"""
        self.learning_rate = max(0.01, min(0.5, rate))
    
    def get_metrics(self) -> Dict:
        """Get RLHF metrics"""
        correct = sum(1 for f in self.feedback_history if f.is_correct)
        total = len(self.feedback_history)
        
        return {
            'totalFeedback': total,
            'correctPredictions': correct,
            'incorrectPredictions': total - correct,
            'accuracyRate': correct / total if total > 0 else 0,
            'weightAdjustments': len(self.weight_history),
            'lastUpdate': self.weight_history[-1].timestamp.isoformat() if self.weight_history else None
        }
    
    def get_feedback_history(self, limit: int = 50) -> List[Dict]:
        """Get feedback history"""
        return [f.to_dict() for f in self.feedback_history[-limit:]]
    
    def get_weight_history(self) -> List[Dict]:
        """Get weight adjustment history"""
        return [h.to_dict() for h in self.weight_history]
    
    def reset_weights(self):
        """Reset to default weights"""
        self.current_weights = EnsembleWeights(
            isolation_forest=settings.DEFAULT_ISOLATION_FOREST_WEIGHT,
            autoencoder=settings.DEFAULT_AUTOENCODER_WEIGHT,
            kmeans=settings.DEFAULT_KMEANS_WEIGHT,
            knn=settings.DEFAULT_KNN_WEIGHT
        )
        
        self.weight_history.append(WeightHistory(
            weights=self.current_weights,
            reason="Manual reset to defaults"
        ))
        
        self._save_data()
    
    def _save_data(self):
        """Save data to disk"""
        data = {
            'feedback': [f.to_dict() for f in self.feedback_history],
            'weights': self.current_weights.to_dict(),
            'history': [h.to_dict() for h in self.weight_history]
        }
        
        with open(self._data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_data(self):
        """Load data from disk"""
        if self._data_path.exists():
            try:
                with open(self._data_path, 'r') as f:
                    data = json.load(f)
                
                # Load weights
                if 'weights' in data:
                    self.current_weights = EnsembleWeights.from_dict(data['weights'])
                
                logger.info(f"Loaded RLHF data from {self._data_path}")
            except Exception as e:
                logger.warning(f"Failed to load RLHF data: {e}")
    
    def export_data(self) -> Dict:
        """Export all data"""
        return {
            'feedback': [f.to_dict() for f in self.feedback_history],
            'weights': self.current_weights.to_dict(),
            'history': [h.to_dict() for h in self.weight_history]
        }
    
    def import_data(self, data: Dict):
        """Import data"""
        if 'weights' in data:
            self.current_weights = EnsembleWeights.from_dict(data['weights'])
        
        self._save_data()


# Singleton instance
rlhf_service = RLHFService()
