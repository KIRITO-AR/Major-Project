"""
Detection Service
Central service for handling intrusion detection
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import numpy as np

from ..ml import EnsembleDetector, extract_features, TrainingDataGenerator
from ..config import settings

logger = logging.getLogger(__name__)


class DetectionService:
    """Central detection service"""
    
    def __init__(self):
        self.detector: Optional[EnsembleDetector] = None
        self._is_initialized = False
        self._data_generator = TrainingDataGenerator()
    
    def initialize(self, force_retrain: bool = False) -> EnsembleDetector:
        """Initialize the detector with training data"""
        if self.detector is not None and self._is_initialized and not force_retrain:
            return self.detector
        
        logger.info("Initializing ensemble detector...")
        self.detector = EnsembleDetector()
        
        # Try to load saved models
        if not force_retrain and self.detector.load_all():
            logger.info("Loaded pre-trained models")
            self._is_initialized = True
            return self.detector
        
        # Generate training data and train
        logger.info("Generating training data...")
        X, y, attack_types = self._data_generator.generate_feature_matrix(
            n_samples=1000,
            anomaly_ratio=0.2
        )
        
        logger.info("Training ensemble models...")
        self.detector.fit(X, list(y.astype(bool)), attack_types)
        
        # Save trained models
        self.detector.save_all()
        
        self._is_initialized = True
        return self.detector
    
    def get_detector(self) -> EnsembleDetector:
        """Get the current detector instance"""
        if self.detector is None:
            self.initialize()
        return self.detector
    
    def retrain(self, X: np.ndarray = None, y: List[bool] = None, attack_types: List[str] = None):
        """Reset and retrain the detector"""
        if X is None:
            X, y_arr, attack_types = self._data_generator.generate_feature_matrix(
                n_samples=1000,
                anomaly_ratio=0.2
            )
            y = list(y_arr.astype(bool))
        
        self.detector = EnsembleDetector()
        self.detector.fit(X, y, attack_types)
        self.detector.save_all()
        self._is_initialized = True
    
    def detect_anomaly(
        self, 
        packet: Dict,
        method: str = 'Ensemble'
    ) -> Dict[str, Any]:
        """
        Detect anomaly in a network packet
        
        Args:
            packet: Network packet dictionary
            method: Detection method to use
            
        Returns:
            Detection result dictionary
        """
        detector = self.get_detector()
        features = extract_features(packet)
        
        # Get prediction
        if method == 'Ensemble':
            prediction = detector.predict(features)
            score = prediction.score
            is_anomaly = prediction.is_anomaly
            attack_type = prediction.attack_type
            model_scores = prediction.scores
        else:
            score = detector.predict_by_method(features, method)
            is_anomaly = score > detector.anomaly_threshold
            attack_type = None
            model_scores = {
                'isolation_forest': detector.predict_by_method(features, 'Isolation Forest'),
                'autoencoder': detector.predict_by_method(features, 'Autoencoder'),
                'kmeans': detector.predict_by_method(features, 'K-Means'),
                'knn': detector.predict_by_method(features, 'KNN')
            }
        
        # Determine threat level
        threat_level = self._get_threat_level(score)
        
        # Classify attack type if anomaly
        if is_anomaly and attack_type is None:
            attack_type = self._classify_attack(packet, score)
        
        # Generate description and recommendations
        description = self._generate_description(is_anomaly, attack_type, packet)
        recommendations = self._generate_recommendations(attack_type, threat_level)
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'packet': packet,
            'isAnomaly': is_anomaly,
            'threatLevel': threat_level,
            'attackType': attack_type,
            'confidence': round(score * 100, 2),
            'detectionMethod': method,
            'description': description,
            'recommendations': recommendations,
            'modelScores': {
                'isolationForest': round(model_scores.get('isolation_forest', 0), 4),
                'autoencoder': round(model_scores.get('autoencoder', 0), 4),
                'kMeans': round(model_scores.get('kmeans', 0), 4),
                'knn': round(model_scores.get('knn', 0), 4)
            }
        }
    
    def detect_batch(
        self, 
        packets: List[Dict],
        method: str = 'Ensemble'
    ) -> List[Dict]:
        """Batch detection for multiple packets"""
        return [self.detect_anomaly(packet, method) for packet in packets]
    
    def _get_threat_level(self, score: float) -> str:
        """Get threat level from score"""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.75:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        return 'low'
    
    def _classify_attack(self, packet: Dict, score: float) -> Optional[str]:
        """Classify the attack type based on packet characteristics"""
        dest_port = packet.get('destPort', packet.get('dest_port', 0))
        protocol = packet.get('protocol', 'TCP')
        packet_size = packet.get('packetSize', packet.get('packet_size', 0))
        flags = packet.get('flags', '')
        
        # Simple heuristic classification
        if dest_port in [22, 3389] and packet_size < 500:
            return 'Brute Force'
        elif dest_port in [80, 443, 8080] and packet_size > 1000:
            if 'sql' in str(packet.get('payload', '')).lower():
                return 'SQL Injection'
            return 'XSS'
        elif packet_size < 100 and protocol == 'ICMP':
            return 'Probe'
        elif flags and any(f in flags.upper() for f in ['SYN', 'FIN', 'RST']):
            if score > 0.85:
                return 'DDoS'
            return 'DoS'
        elif dest_port < 1024:
            return 'Port Scan'
        
        return 'Unknown'
    
    def _generate_description(
        self, 
        is_anomaly: bool, 
        attack_type: Optional[str],
        packet: Dict
    ) -> str:
        """Generate detection description"""
        source_ip = packet.get('sourceIP', packet.get('source_ip', 'Unknown'))
        dest_ip = packet.get('destIP', packet.get('dest_ip', 'Unknown'))
        
        if not is_anomaly:
            return f"Normal traffic detected from {source_ip} to {dest_ip}"
        
        attack_descriptions = {
            'DoS': 'Denial of Service attack pattern detected',
            'DDoS': 'Distributed Denial of Service attack detected',
            'Port Scan': 'Port scanning activity detected',
            'Brute Force': 'Brute force authentication attempt detected',
            'SQL Injection': 'SQL injection attempt detected',
            'XSS': 'Cross-site scripting attempt detected',
            'Probe': 'Network reconnaissance activity detected',
            'Malware': 'Malware communication pattern detected',
            'Unknown': 'Suspicious network activity detected'
        }
        
        return f"{attack_descriptions.get(attack_type, 'Anomaly detected')} from {source_ip}"
    
    def _generate_recommendations(
        self, 
        attack_type: Optional[str],
        threat_level: str
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if threat_level in ['critical', 'high']:
            recommendations.append('Consider blocking the source IP immediately')
            recommendations.append('Alert security team for investigation')
        
        attack_recommendations = {
            'DoS': ['Implement rate limiting', 'Enable DDoS protection'],
            'DDoS': ['Activate DDoS mitigation', 'Contact ISP for upstream filtering'],
            'Port Scan': ['Review firewall rules', 'Enable port scan detection'],
            'Brute Force': ['Implement account lockout', 'Enable multi-factor authentication'],
            'SQL Injection': ['Review input validation', 'Use parameterized queries'],
            'XSS': ['Enable Content Security Policy', 'Sanitize user inputs'],
            'Probe': ['Monitor for follow-up attacks', 'Review exposed services'],
            'Malware': ['Isolate affected systems', 'Run antivirus scan']
        }
        
        if attack_type in attack_recommendations:
            recommendations.extend(attack_recommendations[attack_type])
        
        if not recommendations:
            recommendations.append('Continue monitoring network traffic')
        
        return recommendations
    
    def generate_test_packets(self, count: int = 10) -> List[Dict]:
        """Generate test packets for simulation"""
        packets = []
        for _ in range(count):
            if np.random.random() < 0.3:  # 30% anomalies
                attack_type = np.random.choice(list(self._data_generator.ATTACK_PATTERNS.keys()) if hasattr(self._data_generator, 'ATTACK_PATTERNS') else ['DoS', 'DDoS', 'Port Scan'])
                packet = self._data_generator.generate_attack_packet(attack_type)
            else:
                packet = self._data_generator.generate_normal_packet()
            packets.append(packet)
        return packets
    
    def get_weights(self) -> Dict[str, float]:
        """Get current ensemble weights"""
        detector = self.get_detector()
        return detector.get_weights().to_dict()
    
    def update_weights(self, weights: Dict[str, float]):
        """Update ensemble weights"""
        detector = self.get_detector()
        detector.update_weights(weights)


# Singleton instance
detection_service = DetectionService()
