"""
Training Data Generation
Generates synthetic training data for ML models
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from datetime import datetime, timedelta
import uuid


# Attack patterns with characteristic features
ATTACK_PATTERNS = {
    'DoS': {
        'port_range': (80, 443),
        'packet_size_range': (40, 100),
        'protocol': ['TCP', 'UDP'],
        'high_frequency': True
    },
    'DDoS': {
        'port_range': (80, 8080),
        'packet_size_range': (40, 150),
        'protocol': ['TCP', 'UDP', 'ICMP'],
        'high_frequency': True,
        'multiple_sources': True
    },
    'Port Scan': {
        'port_range': (1, 65535),
        'packet_size_range': (40, 60),
        'protocol': ['TCP'],
        'sequential_ports': True
    },
    'Brute Force': {
        'port_range': (22, 3389),
        'packet_size_range': (100, 500),
        'protocol': ['TCP', 'SSH'],
        'repeated_auth': True
    },
    'SQL Injection': {
        'port_range': (80, 443),
        'packet_size_range': (500, 2000),
        'protocol': ['HTTP', 'HTTPS'],
        'malicious_payload': True
    },
    'XSS': {
        'port_range': (80, 443),
        'packet_size_range': (200, 1500),
        'protocol': ['HTTP', 'HTTPS'],
        'script_payload': True
    },
    'Probe': {
        'port_range': (1, 1024),
        'packet_size_range': (40, 100),
        'protocol': ['ICMP', 'TCP', 'UDP'],
        'reconnaissance': True
    },
    'Malware': {
        'port_range': (1024, 65535),
        'packet_size_range': (500, 5000),
        'protocol': ['TCP', 'HTTP'],
        'encrypted_payload': True
    }
}

NORMAL_PROTOCOLS = ['HTTP', 'HTTPS', 'DNS', 'TCP', 'UDP']
ATTACK_PROTOCOLS = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'SSH', 'FTP']


class TrainingDataGenerator:
    """Generates synthetic training data for IDS"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_normal_packet(self) -> Dict:
        """Generate a normal network packet"""
        protocols = NORMAL_PROTOCOLS
        protocol = random.choice(protocols)
        
        # Normal port ranges
        source_port = random.randint(1024, 65535)
        dest_port = random.choice([80, 443, 53, 22, 21, 25, 110, 143])
        
        # Normal packet sizes
        packet_size = random.randint(64, 1500)
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'sourceIP': self._generate_internal_ip(),
            'destIP': self._generate_external_ip(),
            'sourcePort': source_port,
            'destPort': dest_port,
            'protocol': protocol,
            'packetSize': packet_size,
            'flags': random.choice(['SYN', 'ACK', 'SYN-ACK', '']),
            'is_anomaly': False,
            'attack_type': None
        }
    
    def generate_attack_packet(self, attack_type: Optional[str] = None) -> Dict:
        """Generate an attack packet"""
        if attack_type is None:
            attack_type = random.choice(list(ATTACK_PATTERNS.keys()))
        
        pattern = ATTACK_PATTERNS.get(attack_type, ATTACK_PATTERNS['DoS'])
        
        # Attack-specific features
        source_port = random.randint(1, 65535)
        dest_port = random.randint(*pattern['port_range'])
        protocol = random.choice(pattern['protocol'])
        packet_size = random.randint(*pattern['packet_size_range'])
        
        # Suspicious flags for attacks
        if attack_type in ['DoS', 'DDoS', 'Port Scan']:
            flags = random.choice(['SYN', 'FIN', 'RST', 'SYN-FIN', 'URG'])
        else:
            flags = random.choice(['SYN', 'ACK', ''])
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'sourceIP': self._generate_attacker_ip(),
            'destIP': self._generate_internal_ip(),
            'sourcePort': source_port,
            'destPort': dest_port,
            'protocol': protocol,
            'packetSize': packet_size,
            'flags': flags,
            'is_anomaly': True,
            'attack_type': attack_type
        }
    
    def _generate_internal_ip(self) -> str:
        """Generate internal IP address"""
        return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def _generate_external_ip(self) -> str:
        """Generate external IP address"""
        return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def _generate_attacker_ip(self) -> str:
        """Generate attacker IP address"""
        # Use suspicious IP ranges
        prefixes = ['45.', '185.', '91.', '5.', '46.']
        prefix = random.choice(prefixes)
        return f"{prefix}{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    
    def generate_dataset(
        self, 
        n_samples: int = 1000, 
        anomaly_ratio: float = 0.2
    ) -> Tuple[List[Dict], List[bool], List[Optional[str]]]:
        """
        Generate a mixed dataset of normal and attack traffic
        
        Returns:
            packets: List of packet dictionaries
            labels: List of boolean labels (True = anomaly)
            attack_types: List of attack types (None for normal)
        """
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies
        
        packets = []
        labels = []
        attack_types = []
        
        # Generate normal packets
        for _ in range(n_normal):
            packet = self.generate_normal_packet()
            packets.append(packet)
            labels.append(False)
            attack_types.append(None)
        
        # Generate attack packets
        for _ in range(n_anomalies):
            attack_type = random.choice(list(ATTACK_PATTERNS.keys()))
            packet = self.generate_attack_packet(attack_type)
            packets.append(packet)
            labels.append(True)
            attack_types.append(attack_type)
        
        # Shuffle
        combined = list(zip(packets, labels, attack_types))
        random.shuffle(combined)
        packets, labels, attack_types = zip(*combined)
        
        return list(packets), list(labels), list(attack_types)
    
    def generate_feature_matrix(
        self, 
        n_samples: int = 1000, 
        anomaly_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
        """
        Generate feature matrix for training
        
        Returns:
            X: Feature matrix (n_samples, 7)
            y: Labels array
            attack_types: Attack type strings
        """
        from .features import extract_features
        
        packets, labels, attack_types = self.generate_dataset(n_samples, anomaly_ratio)
        
        X = np.array([extract_features(p) for p in packets])
        y = np.array(labels)
        
        return X, y, attack_types
