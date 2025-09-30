"""
High-level orchestration: a class that ties discovery + fetching + caching.
Provides a clean API for notebooks and the CLI.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Iterable
from tqdm.auto import tqdm
import sys
from .discovery import list_game_ids_for_season as _discover
from .fetch import fetch_and_cache_pbp as _fetch
from .cache import write_manifest_csv
from .config import SHOW_PROGRESS

def _maybe_tqdm(it: Iterable, enable: bool, total: Optional[int] = None, desc: Optional[str] = None):
    if enable:
        return tqdm(it, total=total, desc=desc, leave=False, file=sys.stderr)
    return it

class NHLPBPDownloader:
    """
    Download and cache NHL play-by-play using the new NHL web API.

    Methods
    -------
    list_game_ids_for_season(y, include_regular=True, include_playoffs=True, progress=SHOW_PROGRESS) -> List[int]
    fetch_and_cache_pbp(game_id: int, force=False) -> Dict[str, Any]
    download_season(y, include_regular=True, include_playoffs=True, limit=None, progress=SHOW_PROGRESS) -> List[int]
    write_manifest(y, out_csv_path) -> int
    """

    def list_game_ids_for_season(self, season_start_year: int,
                                 include_regular: bool = True,
                                 include_playoffs: bool = True,
                                 progress: bool = SHOW_PROGRESS) -> List[int]:
        return _discover(season_start_year, include_regular, include_playoffs, progress)

    def fetch_and_cache_pbp(self, game_id: int, force: bool = False) -> Dict[str, Any]:
        return _fetch(game_id, force=force)

    def download_season(self, season_start_year: int,
                        include_regular: bool = True,
                        include_playoffs: bool = True,
                        limit: Optional[int] = None,
                        progress: bool = SHOW_PROGRESS) -> List[int]:
        ids = self.list_game_ids_for_season(season_start_year, include_regular, include_playoffs, progress)
        if limit is not None:
            ids = ids[:limit]
        for gid in _maybe_tqdm(ids, progress, total=len(ids), desc=f"{season_start_year}-{season_start_year+1} downloads"):
            try:
                self.fetch_and_cache_pbp(gid)
            except Exception as e:
                print(f"[warn] game {gid}: {e}")
        return ids

    def write_manifest(self, season_start_year: int, out_csv_path: str) -> int:
        return write_manifest_csv(season_start_year, out_csv_path)