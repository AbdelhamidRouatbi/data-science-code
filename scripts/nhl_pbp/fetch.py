"""
Download play-by-play JSON for a given game ID (new API).
"""

from __future__ import annotations
from typing import Dict, Any
import time, os
from .config import API_BASE, REQUEST_PAUSE_SEC
from .http import get_json
from .cache import cache_path_for_game, write_json, read_json

def fetch_and_cache_pbp(game_id: int, force: bool = False) -> Dict[str, Any]:
    """
    Endpoint: /v1/gamecenter/{game_id}/play-by-play
    If cache exists and force=False, read it; else request and write.
    AI-DOCSTRING: Drafted with AI.
    """
    path = cache_path_for_game(game_id)
    if not force and os.path.exists(path):
        return read_json(path)
    url = f"{API_BASE}/gamecenter/{game_id}/play-by-play"
    data = get_json(url)
    write_json(path, data)
    time.sleep(REQUEST_PAUSE_SEC)
    return data