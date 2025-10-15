"""
HTTP helper: single robust GET with retries & small backoff.
AI-DOCSTRING: Drafted with AI.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Optional
import requests
from .config import TIMEOUT_SEC, MAX_RETRIES

def get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    last = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i == MAX_RETRIES - 1:
                break
            time.sleep(1.2)  # small backoff
    raise RuntimeError(f"Failed after {MAX_RETRIES} tries: {url}\nLast error: {last}")