"""
Cache utilities: path conventions, read/write JSON, manifest.
We organize by season folder to keep directories small.
"""

from __future__ import annotations
import os, json, pathlib, time
from typing import Dict, Any, Iterable, List, Tuple
from .config import CACHE_DIR
from .constants import season_str

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def season_folder(season_start_year: int) -> str:
    folder = os.path.join(CACHE_DIR, f"{season_start_year}-{season_start_year+1}")
    ensure_dir(folder)
    return folder

def cache_path_for_game(game_id: int) -> str:
    # First 4 digits of game id are the season start year (e.g., 2016 from 20162017xxxx)
    season_start = int(str(game_id)[:4])
    return os.path.join(season_folder(season_start), f"{game_id}.json")

def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_cached_games(season_start_year: int) -> Iterable[Tuple[int, str]]:
    """
    Yield (game_id, path) for all files in that season folder.
    AI-DOCSTRING: Drafted with AI.
    """
    folder = season_folder(season_start_year)
    for p in pathlib.Path(folder).glob("*.json"):
        name = p.stem
        if name.isdigit():
            yield int(name), str(p)

def write_manifest_csv(season_start_year: int, out_path: str) -> int:
    """
    Create a simple manifest CSV for a season: game_id,path,bytes,modified_epoch
    Returns number of rows written.
    AI-DOCSTRING: Drafted with AI.
    """
    import csv, os
    rows = []
    for gid, path in iter_cached_games(season_start_year):
        st = os.stat(path)
        rows.append([gid, path, st.st_size, int(st.st_mtime)])
    rows.sort(key=lambda r: r[0])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_id","path","bytes","modified_epoch"])
        w.writerows(rows)
    return len(rows)