"""
Detection API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from ..services import detection_service, auto_response_service, auto_training_service

router = APIRouter(prefix="/detect", tags=["Detection"])


class DetectRequest(BaseModel):
    """Detection request model"""
    count: int = 10
    method: str = "Ensemble"
    packets: Optional[List[Dict[str, Any]]] = None


class PacketDetectRequest(BaseModel):
    """Single packet detection request"""
    packet: Dict[str, Any]
    method: str = "Ensemble"


@router.post("")
async def detect(request: DetectRequest):
    """
    Run detection on network packets
    
    If packets are provided, use those; otherwise generate test packets
    """
    try:
        # Initialize detector if needed
        detection_service.initialize()
        
        # Get or generate packets
        if request.packets:
            packets = request.packets
        else:
            packets = detection_service.generate_test_packets(request.count)
        
        # Run detection
        results = detection_service.detect_batch(packets, request.method)
        
        # Evaluate auto-response for each result
        for result in results:
            response_action = auto_response_service.evaluate_threat(result)
            result['autoResponseAction'] = response_action['action']
            
            # Add to training data
            auto_training_service.add_detection_data(result)
        
        # Calculate summary
        anomalies = [r for r in results if r['isAnomaly']]
        summary = {
            'total': len(results),
            'anomalies': len(anomalies),
            'normal': len(results) - len(anomalies),
            'critical': sum(1 for r in results if r['threatLevel'] == 'critical'),
            'high': sum(1 for r in results if r['threatLevel'] == 'high'),
            'medium': sum(1 for r in results if r['threatLevel'] == 'medium'),
            'low': sum(1 for r in results if r['threatLevel'] == 'low'),
            'blocked': sum(1 for r in results if r.get('autoResponseAction') == 'block')
        }
        
        return {
            'results': results,
            'summary': summary,
            'weights': detection_service.get_weights()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/single")
async def detect_single(request: PacketDetectRequest):
    """Detect anomaly in a single packet"""
    try:
        detection_service.initialize()
        
        result = detection_service.detect_anomaly(request.packet, request.method)
        
        # Evaluate auto-response
        response_action = auto_response_service.evaluate_threat(result)
        result['autoResponseAction'] = response_action['action']
        
        # Add to training data
        auto_training_service.add_detection_data(result)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """Get detector status"""
    try:
        detector = detection_service.get_detector()
        
        return {
            'initialized': detector is not None,
            'trained': detector.is_trained() if detector else False,
            'weights': detection_service.get_weights(),
            'autoResponse': auto_response_service.get_stats(),
            'training': auto_training_service.get_stats()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def retrain():
    """Force detector retraining"""
    try:
        detection_service.retrain()
        
        return {
            'success': True,
            'message': 'Detector retrained successfully'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
