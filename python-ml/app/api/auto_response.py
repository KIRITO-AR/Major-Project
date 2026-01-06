"""
Auto-Response API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from ..services import auto_response_service

router = APIRouter(prefix="/auto-response", tags=["Auto-Response"])


class BlockRequest(BaseModel):
    """Block IP request"""
    ip_address: str = Field(..., alias="ipAddress")
    reason: str
    attack_type: Optional[str] = Field(None, alias="attackType")
    duration: Optional[int] = None  # minutes
    
    class Config:
        populate_by_name = True


class UnblockRequest(BaseModel):
    """Unblock IP request"""
    ip_address: str = Field(..., alias="ipAddress")
    reason: Optional[str] = "Manual unblock"
    
    class Config:
        populate_by_name = True


class ConfigUpdateRequest(BaseModel):
    """Config update request"""
    enabled: Optional[bool] = None
    threat_threshold: Optional[float] = Field(None, alias="threatThreshold")
    auto_block_duration: Optional[int] = Field(None, alias="autoBlockDuration")
    block_on_critical: Optional[bool] = Field(None, alias="blockOnCritical")
    block_on_high: Optional[bool] = Field(None, alias="blockOnHigh")
    block_on_medium: Optional[bool] = Field(None, alias="blockOnMedium")
    
    class Config:
        populate_by_name = True


class WhitelistRequest(BaseModel):
    """Whitelist request"""
    ip_address: str = Field(..., alias="ipAddress")
    
    class Config:
        populate_by_name = True


@router.get("")
async def get_status():
    """Get auto-response status"""
    return {
        'config': auto_response_service.get_config(),
        'stats': auto_response_service.get_stats(),
        'blockedIPs': [b.to_dict() for b in auto_response_service.get_blocked_ips()]
    }


@router.get("/config")
async def get_config():
    """Get auto-response configuration"""
    return auto_response_service.get_config()


@router.post("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update auto-response configuration"""
    try:
        updates = {}
        
        if request.enabled is not None:
            updates['enabled'] = request.enabled
        if request.threat_threshold is not None:
            updates['threatThreshold'] = request.threat_threshold
        if request.auto_block_duration is not None:
            updates['autoBlockDuration'] = request.auto_block_duration
        if request.block_on_critical is not None:
            updates['blockOnCritical'] = request.block_on_critical
        if request.block_on_high is not None:
            updates['blockOnHigh'] = request.block_on_high
        if request.block_on_medium is not None:
            updates['blockOnMedium'] = request.block_on_medium
        
        config = auto_response_service.update_config(updates)
        
        return {
            'success': True,
            'config': config
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/block")
async def block_ip(request: BlockRequest):
    """Block an IP address"""
    try:
        blocked = auto_response_service.block_ip(
            ip_address=request.ip_address,
            reason=request.reason,
            attack_type=request.attack_type,
            auto_blocked=False,
            duration=request.duration
        )
        
        return {
            'success': True,
            'blocked': blocked.to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unblock")
async def unblock_ip(request: UnblockRequest):
    """Unblock an IP address"""
    try:
        success = auto_response_service.unblock_ip(
            ip_address=request.ip_address,
            reason=request.reason or "Manual unblock"
        )
        
        return {
            'success': success,
            'message': 'IP unblocked' if success else 'IP was not blocked'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocked")
async def get_blocked_ips():
    """Get list of blocked IPs"""
    return {
        'blockedIPs': [b.to_dict() for b in auto_response_service.get_blocked_ips()]
    }


@router.get("/events")
async def get_events(limit: int = 50):
    """Get block events history"""
    return {
        'events': auto_response_service.get_block_events(limit)
    }


@router.post("/whitelist/add")
async def add_to_whitelist(request: WhitelistRequest):
    """Add IP to whitelist"""
    try:
        auto_response_service.add_to_whitelist(request.ip_address)
        
        return {
            'success': True,
            'message': f'{request.ip_address} added to whitelist'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/whitelist/remove")
async def remove_from_whitelist(request: WhitelistRequest):
    """Remove IP from whitelist"""
    try:
        auto_response_service.remove_from_whitelist(request.ip_address)
        
        return {
            'success': True,
            'message': f'{request.ip_address} removed from whitelist'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get auto-response statistics"""
    return auto_response_service.get_stats()
