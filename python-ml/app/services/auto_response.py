"""
Auto-Response Service
Automatic attack prevention without human intervention
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from ..config import settings

logger = logging.getLogger(__name__)


class BlockedIP:
    """Blocked IP entry"""
    def __init__(
        self,
        ip_address: str,
        reason: str,
        attack_type: Optional[str] = None,
        confidence: float = 0,
        auto_blocked: bool = False,
        duration: Optional[int] = None  # minutes, None = permanent
    ):
        self.id = str(uuid.uuid4())
        self.ip_address = ip_address
        self.reason = reason
        self.attack_type = attack_type
        self.confidence = confidence
        self.auto_blocked = auto_blocked
        self.blocked_at = datetime.now()
        
        if duration and duration > 0:
            self.expires_at = datetime.now() + timedelta(minutes=duration)
        else:
            self.expires_at = None
    
    def is_expired(self) -> bool:
        """Check if block has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'ipAddress': self.ip_address,
            'reason': self.reason,
            'attackType': self.attack_type,
            'confidence': self.confidence,
            'blockedAt': self.blocked_at.isoformat(),
            'expiresAt': self.expires_at.isoformat() if self.expires_at else None,
            'autoBlocked': self.auto_blocked
        }


class BlockEvent:
    """Block event log entry"""
    def __init__(
        self,
        ip_address: str,
        action: str,  # 'blocked', 'unblocked', 'extended'
        reason: str,
        auto_triggered: bool = False
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        self.ip_address = ip_address
        self.action = action
        self.reason = reason
        self.auto_triggered = auto_triggered
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'ipAddress': self.ip_address,
            'action': self.action,
            'reason': self.reason,
            'autoTriggered': self.auto_triggered
        }


class AutoResponseConfig:
    """Auto-response configuration"""
    def __init__(self):
        self.enabled = settings.AUTO_RESPONSE_ENABLED
        self.threat_threshold = settings.AUTO_RESPONSE_THREAT_THRESHOLD
        self.auto_block_duration = settings.AUTO_RESPONSE_BLOCK_DURATION
        self.block_on_critical = True
        self.block_on_high = True
        self.block_on_medium = False
        self.notify_on_block = True
        self.whitelisted_ips = ['127.0.0.1', 'localhost', '::1']
    
    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'threatThreshold': self.threat_threshold,
            'autoBlockDuration': self.auto_block_duration,
            'blockOnCritical': self.block_on_critical,
            'blockOnHigh': self.block_on_high,
            'blockOnMedium': self.block_on_medium,
            'notifyOnBlock': self.notify_on_block,
            'whitelistedIPs': self.whitelisted_ips
        }
    
    def update(self, data: Dict):
        """Update configuration from dictionary"""
        if 'enabled' in data:
            self.enabled = data['enabled']
        if 'threatThreshold' in data:
            self.threat_threshold = data['threatThreshold']
        if 'autoBlockDuration' in data:
            self.auto_block_duration = data['autoBlockDuration']
        if 'blockOnCritical' in data:
            self.block_on_critical = data['blockOnCritical']
        if 'blockOnHigh' in data:
            self.block_on_high = data['blockOnHigh']
        if 'blockOnMedium' in data:
            self.block_on_medium = data['blockOnMedium']
        if 'notifyOnBlock' in data:
            self.notify_on_block = data['notifyOnBlock']
        if 'whitelistedIPs' in data:
            self.whitelisted_ips = data['whitelistedIPs']


class AutoResponseService:
    """Auto-response service for automatic threat mitigation"""
    
    def __init__(self):
        self.blocked_ips: Dict[str, BlockedIP] = {}
        self.block_events: List[BlockEvent] = []
        self.config = AutoResponseConfig()
        self._data_path = settings.DATA_DIR / "auto_response_data.json"
        
        # Load saved data
        self._load_data()
    
    def evaluate_threat(self, detection: Dict) -> Dict:
        """
        Evaluate a detection and determine response action
        
        Returns:
            {'action': str, 'reason': str, 'autoExecuted': bool}
        """
        if not self.config.enabled:
            return {'action': 'monitor', 'reason': 'Auto-response disabled', 'autoExecuted': False}
        
        packet = detection.get('packet', {})
        source_ip = packet.get('sourceIP', packet.get('source_ip', ''))
        
        # Check whitelist
        if self.is_whitelisted(source_ip):
            return {'action': 'ignore', 'reason': 'IP is whitelisted', 'autoExecuted': False}
        
        # Check if already blocked
        if self.is_blocked(source_ip):
            return {'action': 'ignore', 'reason': 'IP already blocked', 'autoExecuted': False}
        
        # Determine if we should auto-block
        should_block = self._should_auto_block(detection)
        
        if should_block:
            self.block_ip(
                ip_address=source_ip,
                reason=f"Auto-blocked: {detection.get('attackType', 'Anomaly detected')}",
                attack_type=detection.get('attackType'),
                confidence=detection.get('confidence', 0),
                auto_blocked=True,
                duration=self.config.auto_block_duration
            )
            
            return {
                'action': 'block',
                'reason': f"Threat level: {detection.get('threatLevel')}, Confidence: {detection.get('confidence', 0):.1f}%",
                'autoExecuted': True
            }
        
        if detection.get('isAnomaly'):
            return {
                'action': 'alert',
                'reason': 'Anomaly detected but below auto-block threshold',
                'autoExecuted': False
            }
        
        return {'action': 'monitor', 'reason': 'Normal traffic', 'autoExecuted': False}
    
    def _should_auto_block(self, detection: Dict) -> bool:
        """Determine if threat should be auto-blocked"""
        if not detection.get('isAnomaly'):
            return False
        
        confidence = detection.get('confidence', 0)
        if confidence < self.config.threat_threshold * 100:
            return False
        
        threat_level = detection.get('threatLevel', 'low')
        
        if threat_level == 'critical':
            return self.config.block_on_critical
        elif threat_level == 'high':
            return self.config.block_on_high
        elif threat_level == 'medium':
            return self.config.block_on_medium
        
        return False
    
    def block_ip(
        self,
        ip_address: str,
        reason: str,
        attack_type: Optional[str] = None,
        confidence: float = 0,
        auto_blocked: bool = False,
        duration: Optional[int] = None
    ) -> BlockedIP:
        """Block an IP address"""
        blocked = BlockedIP(
            ip_address=ip_address,
            reason=reason,
            attack_type=attack_type,
            confidence=confidence,
            auto_blocked=auto_blocked,
            duration=duration or self.config.auto_block_duration
        )
        
        self.blocked_ips[ip_address] = blocked
        
        self.block_events.append(BlockEvent(
            ip_address=ip_address,
            action='blocked',
            reason=reason,
            auto_triggered=auto_blocked
        ))
        
        self._save_data()
        logger.info(f"Blocked IP: {ip_address} - {reason}")
        
        return blocked
    
    def unblock_ip(self, ip_address: str, reason: str = 'Manual unblock') -> bool:
        """Unblock an IP address"""
        if ip_address not in self.blocked_ips:
            return False
        
        del self.blocked_ips[ip_address]
        
        self.block_events.append(BlockEvent(
            ip_address=ip_address,
            action='unblocked',
            reason=reason,
            auto_triggered=False
        ))
        
        self._save_data()
        logger.info(f"Unblocked IP: {ip_address} - {reason}")
        
        return True
    
    def is_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        if ip_address not in self.blocked_ips:
            return False
        
        blocked = self.blocked_ips[ip_address]
        
        # Check expiration
        if blocked.is_expired():
            del self.blocked_ips[ip_address]
            self.block_events.append(BlockEvent(
                ip_address=ip_address,
                action='unblocked',
                reason='Block expired',
                auto_triggered=True
            ))
            self._save_data()
            return False
        
        return True
    
    def is_whitelisted(self, ip_address: str) -> bool:
        """Check if IP is whitelisted"""
        return ip_address in self.config.whitelisted_ips
    
    def get_blocked_ips(self) -> List[BlockedIP]:
        """Get all blocked IPs (cleaning up expired ones)"""
        # Clean up expired blocks
        expired = [ip for ip, blocked in self.blocked_ips.items() if blocked.is_expired()]
        for ip in expired:
            del self.blocked_ips[ip]
            self.block_events.append(BlockEvent(
                ip_address=ip,
                action='unblocked',
                reason='Block expired',
                auto_triggered=True
            ))
        
        if expired:
            self._save_data()
        
        return list(self.blocked_ips.values())
    
    def get_block_events(self, limit: int = 50) -> List[Dict]:
        """Get block events history"""
        return [e.to_dict() for e in self.block_events[-limit:]]
    
    def update_config(self, updates: Dict) -> Dict:
        """Update configuration"""
        self.config.update(updates)
        self._save_data()
        return self.config.to_dict()
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.to_dict()
    
    def add_to_whitelist(self, ip_address: str):
        """Add IP to whitelist"""
        if ip_address not in self.config.whitelisted_ips:
            self.config.whitelisted_ips.append(ip_address)
        # Also unblock if currently blocked
        self.unblock_ip(ip_address, 'Added to whitelist')
        self._save_data()
    
    def remove_from_whitelist(self, ip_address: str):
        """Remove IP from whitelist"""
        self.config.whitelisted_ips = [ip for ip in self.config.whitelisted_ips if ip != ip_address]
        self._save_data()
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        blocked = self.get_blocked_ips()
        return {
            'totalBlocked': len(blocked),
            'autoBlocked': sum(1 for b in blocked if b.auto_blocked),
            'manualBlocked': sum(1 for b in blocked if not b.auto_blocked),
            'totalEvents': len(self.block_events)
        }
    
    def _save_data(self):
        """Save data to disk"""
        data = {
            'blockedIPs': [b.to_dict() for b in self.blocked_ips.values()],
            'events': [e.to_dict() for e in self.block_events],
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
                
                if 'config' in data:
                    self.config.update(data['config'])
                
                logger.info(f"Loaded auto-response data from {self._data_path}")
            except Exception as e:
                logger.warning(f"Failed to load auto-response data: {e}")
    
    def export_data(self) -> Dict:
        """Export all data"""
        return {
            'blockedIPs': [b.to_dict() for b in self.blocked_ips.values()],
            'events': [e.to_dict() for e in self.block_events],
            'config': self.config.to_dict()
        }


# Singleton instance
auto_response_service = AutoResponseService()
