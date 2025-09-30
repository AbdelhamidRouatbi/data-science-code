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
    """
    path = cache_path_for_game(game_id)
    if not force and os.path.exists(path):
        return read_json(path)
    url = f"{API_BASE}/gamecenter/{game_id}/play-by-play"
    data = get_json(url)
    write_json(path, data)
    time.sleep(REQUEST_PAUSE_SEC)
    return data

# """
# Download play-by-play JSON for a given game ID (legacy Stats API).
# """

# from __future__ import annotations
# from typing import Dict, Any
# import time, os

# from .config import REQUEST_PAUSE_SEC
# from .http import get_json
# from .cache import cache_path_for_game, write_json, read_json

# # Legacy endpoint required by the assignment
# OLD_STATS_BASE = "https://statsapi.web.nhl.com/api/v1/game/{gid}/feed/live"

# def fetch_and_cache_pbp(game_id: int, force: bool = False) -> Dict[str, Any]:
#     """
#     Endpoint (legacy): https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live
#     If cache exists and force=False, read it; else request and write.
#     """
#     path = cache_path_for_game(game_id)
#     if not force and os.path.exists(path):
#         return read_json(path)

#     url = OLD_STATS_BASE.format(gid=game_id)
#     data = get_json(url)

#     # Light sanity checks (won't block caching if missing)
#     try:
#         pk = str(data.get("gameData", {}).get("game", {}).get("pk") or "")
#         if pk and pk != str(game_id):
#             raise ValueError(f"Downloaded gamePk {pk} does not match requested {game_id}")
#         # Optional: ensure plays exist (legacy feed puts them under liveData.plays)
#         plays = data.get("liveData", {}).get("plays", {})
#         if not isinstance(plays, dict):
#             raise ValueError(f"No plays dict found in legacy feed for {game_id}")
#     except Exception as e:
#         # You can choose to `raise` here if you prefer to fail fast.
#         # For now, warn but still cache what we got to aid debugging.
#         print(f"[warn] validation issue for {game_id}: {e}")

#     write_json(path, data)
#     time.sleep(REQUEST_PAUSE_SEC)
#     return data