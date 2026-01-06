"""
Utility helpers
"""
import uuid
from datetime import datetime
from typing import Any, Dict


def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Get current time as ISO string"""
    return datetime.now().isoformat()


def safe_get(d: Dict, *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))
