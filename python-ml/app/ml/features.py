"""
Feature Extraction for Network Packets
"""
import numpy as np
from typing import Dict, List, Union
import hashlib


def normalize_port(port: int) -> float:
    """Normalize port number to 0-1 range"""
    # Well-known ports (0-1023) are lower risk
    # Registered ports (1024-49151) medium
    # Dynamic/private ports (49152-65535) higher risk
    if port <= 1023:
        return port / 1023 * 0.3
    elif port <= 49151:
        return 0.3 + ((port - 1024) / (49151 - 1024)) * 0.4
    else:
        return 0.7 + ((port - 49152) / (65535 - 49152)) * 0.3


def normalize_protocol(protocol: str) -> float:
    """Convert protocol to numeric value"""
    protocol_map = {
        'ICMP': 0.1,
        'DNS': 0.2,
        'HTTP': 0.3,
        'HTTPS': 0.35,
        'FTP': 0.5,
        'SSH': 0.6,
        'TCP': 0.7,
        'UDP': 0.8
    }
    return protocol_map.get(protocol.upper(), 0.5)


def normalize_packet_size(size: int) -> float:
    """Normalize packet size (0-65535 bytes)"""
    # Most normal packets are under 1500 bytes
    if size <= 1500:
        return size / 1500 * 0.5
    else:
        return 0.5 + min((size - 1500) / 64000, 0.5)


def ip_entropy(ip: str) -> float:
    """Calculate entropy-like score from IP address"""
    # Simple hash-based distribution
    hash_val = int(hashlib.md5(ip.encode()).hexdigest()[:8], 16)
    return (hash_val % 1000) / 1000


def flag_score(flags: str) -> float:
    """Score based on TCP flags"""
    if not flags:
        return 0.5
    
    suspicious_flags = ['SYN', 'FIN', 'RST', 'URG']
    score = 0.0
    flags_upper = flags.upper()
    
    for flag in suspicious_flags:
        if flag in flags_upper:
            score += 0.2
    
    return min(score, 1.0)


def extract_features(packet: Dict) -> List[float]:
    """
    Extract feature vector from network packet
    Returns 7-dimensional feature vector
    """
    source_port = packet.get('sourcePort', packet.get('source_port', 0))
    dest_port = packet.get('destPort', packet.get('dest_port', 0))
    protocol = packet.get('protocol', 'TCP')
    packet_size = packet.get('packetSize', packet.get('packet_size', 0))
    source_ip = packet.get('sourceIP', packet.get('source_ip', '0.0.0.0'))
    dest_ip = packet.get('destIP', packet.get('dest_ip', '0.0.0.0'))
    flags = packet.get('flags', '')
    
    features = [
        normalize_port(source_port),
        normalize_port(dest_port),
        normalize_protocol(protocol),
        normalize_packet_size(packet_size),
        ip_entropy(source_ip),
        ip_entropy(dest_ip),
        flag_score(flags or '')
    ]
    
    return features


def extract_features_batch(packets: List[Dict]) -> np.ndarray:
    """Extract features for multiple packets"""
    return np.array([extract_features(p) for p in packets])
