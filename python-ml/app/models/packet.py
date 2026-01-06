"""
Network Packet Models
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid

ProtocolType = Literal['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP']

class NetworkPacket(BaseModel):
    """Network packet model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    source_ip: str = Field(..., alias="sourceIP")
    dest_ip: str = Field(..., alias="destIP")
    source_port: int = Field(..., alias="sourcePort", ge=0, le=65535)
    dest_port: int = Field(..., alias="destPort", ge=0, le=65535)
    protocol: ProtocolType
    packet_size: int = Field(..., alias="packetSize", ge=0)
    flags: Optional[str] = None
    payload: Optional[str] = None
    
    class Config:
        populate_by_name = True
        
class NetworkFlow(BaseModel):
    """Network flow model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    duration: float
    protocol: str
    src_bytes: int = Field(..., alias="srcBytes")
    dst_bytes: int = Field(..., alias="dstBytes")
    src_packets: int = Field(..., alias="srcPackets")
    dst_packets: int = Field(..., alias="dstPackets")
    src_port: int = Field(..., alias="srcPort")
    dst_port: int = Field(..., alias="dstPort")
    tcp_flags: str = Field(..., alias="tcpFlags")
    flow_start: datetime = Field(..., alias="flowStart")
    flow_end: datetime = Field(..., alias="flowEnd")
    
    class Config:
        populate_by_name = True
