"""
RLHF API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from ..services import rlhf_service, detection_service

router = APIRouter(prefix="/rlhf", tags=["RLHF"])


class FeedbackRequest(BaseModel):
    """Feedback submission request"""
    detection_id: str = Field(..., alias="detectionId")
    is_correct: bool = Field(..., alias="isCorrect")
    correct_label: Optional[str] = Field(None, alias="correctLabel")
    correct_attack_type: Optional[str] = Field(None, alias="correctAttackType")
    model_method: Optional[str] = Field(None, alias="modelMethod")
    notes: Optional[str] = None
    
    class Config:
        populate_by_name = True


class WeightUpdateRequest(BaseModel):
    """Weight update request"""
    isolation_forest: Optional[float] = Field(None, alias="isolationForest")
    autoencoder: Optional[float] = None
    k_means: Optional[float] = Field(None, alias="kMeans")
    knn: Optional[float] = None
    
    class Config:
        populate_by_name = True


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a detection"""
    try:
        feedback = rlhf_service.add_feedback(
            detection_id=request.detection_id,
            is_correct=request.is_correct,
            correct_label=request.correct_label,
            attack_type=request.correct_attack_type,
            model_method=request.model_method,
            notes=request.notes
        )
        
        return {
            'success': True,
            'feedback': feedback.to_dict(),
            'weights': rlhf_service.get_weights().to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weights")
async def get_weights():
    """Get current ensemble weights"""
    return {
        'weights': rlhf_service.get_weights().to_dict()
    }


@router.post("/adjust")
async def adjust_weights():
    """Force weight adjustment based on feedback"""
    try:
        weights = rlhf_service.adjust_weights()
        
        # Update detector weights
        detection_service.update_weights(weights.to_dict())
        
        return {
            'success': True,
            'weights': weights.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_weights():
    """Reset weights to defaults"""
    try:
        rlhf_service.reset_weights()
        weights = rlhf_service.get_weights()
        
        # Update detector weights
        detection_service.update_weights(weights.to_dict())
        
        return {
            'success': True,
            'weights': weights.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get RLHF metrics"""
    return rlhf_service.get_metrics()


@router.get("/history")
async def get_history(limit: int = 50):
    """Get feedback history"""
    return {
        'feedback': rlhf_service.get_feedback_history(limit),
        'weights': rlhf_service.get_weight_history()
    }


@router.get("")
async def get_rlhf_status():
    """Get RLHF status"""
    return {
        'metrics': rlhf_service.get_metrics(),
        'weights': rlhf_service.get_weights().to_dict(),
        'history': rlhf_service.get_weight_history()
    }
