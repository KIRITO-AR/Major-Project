"""
Training API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from ..services import auto_training_service, detection_service

router = APIRouter(prefix="/training", tags=["Training"])


class VerifyRequest(BaseModel):
    """Verify data point request"""
    id: str
    is_correct: bool = Field(..., alias="isCorrect")
    correct_label: Optional[str] = Field(None, alias="correctLabel")
    
    class Config:
        populate_by_name = True


class ImportRequest(BaseModel):
    """Import training data request"""
    data: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """Config update request"""
    enabled: Optional[bool] = None
    min_samples_for_retrain: Optional[int] = Field(None, alias="minSamplesForRetrain")
    auto_retrain_on_new_anomalies: Optional[bool] = Field(None, alias="autoRetrainOnNewAnomalies")
    max_stored_samples: Optional[int] = Field(None, alias="maxStoredSamples")
    include_normal_traffic: Optional[bool] = Field(None, alias="includeNormalTraffic")
    normal_traffic_ratio: Optional[float] = Field(None, alias="normalTrafficRatio")
    
    class Config:
        populate_by_name = True


@router.get("")
async def get_status():
    """Get training status"""
    return {
        'stats': auto_training_service.get_stats(),
        'config': auto_training_service.get_config()
    }


@router.get("/data")
async def get_training_data(
    label: Optional[str] = None,
    verified: Optional[bool] = None,
    limit: int = 100
):
    """Get training data"""
    return {
        'data': auto_training_service.get_training_data(
            label=label,
            verified=verified,
            limit=limit
        ),
        'stats': auto_training_service.get_stats()
    }


@router.post("/verify")
async def verify_data_point(request: VerifyRequest):
    """Verify a training data point"""
    try:
        auto_training_service.verify_data_point(
            id=request.id,
            is_correct=request.is_correct,
            correct_label=request.correct_label
        )
        
        return {
            'success': True,
            'message': 'Data point verified'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def retrain():
    """Trigger model retraining"""
    try:
        result = auto_training_service.execute_retraining()
        
        # Also retrain the detector
        detection_service.retrain()
        
        return {
            'success': True,
            'result': result.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_data():
    """Export training data as JSON"""
    return auto_training_service.export_training_data()


@router.post("/import")
async def import_data(request: ImportRequest):
    """Import training data from JSON"""
    try:
        auto_training_service.import_training_data(request.data)
        
        return {
            'success': True,
            'stats': auto_training_service.get_stats()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_data():
    """Clear all training data"""
    try:
        auto_training_service.clear_training_data()
        
        return {
            'success': True,
            'message': 'Training data cleared'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{sample_id}")
async def delete_sample(sample_id: str):
    """Delete a training sample"""
    try:
        success = auto_training_service.delete_training_sample(sample_id)
        
        return {
            'success': success,
            'message': 'Sample deleted' if success else 'Sample not found'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_config():
    """Get training configuration"""
    return auto_training_service.get_config()


@router.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update training configuration"""
    try:
        updates = {}
        
        if request.enabled is not None:
            updates['enabled'] = request.enabled
        if request.min_samples_for_retrain is not None:
            updates['minSamplesForRetrain'] = request.min_samples_for_retrain
        if request.auto_retrain_on_new_anomalies is not None:
            updates['autoRetrainOnNewAnomalies'] = request.auto_retrain_on_new_anomalies
        if request.max_stored_samples is not None:
            updates['maxStoredSamples'] = request.max_stored_samples
        if request.include_normal_traffic is not None:
            updates['includeNormalTraffic'] = request.include_normal_traffic
        if request.normal_traffic_ratio is not None:
            updates['normalTrafficRatio'] = request.normal_traffic_ratio
        
        config = auto_training_service.update_config(updates)
        
        return {
            'success': True,
            'config': config
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get training statistics"""
    return auto_training_service.get_stats()
