"""
Season-wide discovery of game IDs using the new NHL API:

  GET /v1/club-schedule-season/{TEAM_TRICODE}/{SEASON}

We loop all teams for the season and de-duplicate `id` fields. We then
filter out preseason (PR) and keep regular (R) and playoffs (P).
AI-DOCSTRING: Drafted with AI.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Iterable
import time, sys
from tqdm.auto import tqdm
from .config import API_BASE, REQUEST_PAUSE_SEC, SHOW_PROGRESS
from .http import get_json
from .constants import season_str, tricodes_for_season

def _extract_game_id(item: Any) -> Optional[int]:
    if isinstance(item, dict):
        gid = item.get("id") or item.get("gameId") or item.get("gamePk")
        if isinstance(gid, int):
            return gid
        if isinstance(gid, str) and gid.isdigit():
            return int(gid)
    return None

def _extract_game_type(item: Dict[str, Any]) -> str:
    """
    Return one of {'PR','R','P','A','UNK'} from a schedule item.
    New API isn't fully consistent, so we try multiple fields.
    AI-DOCSTRING: Drafted with AI.
    """
    def normalize(v: Any) -> Optional[str]:
        if isinstance(v, str):
            v = v.upper()
            if v in {"PR","R","P","A"}:
                return v
            if v in {"01","02","03","04"}:
                return {"01":"PR","02":"R","03":"P","04":"A"}[v]
        if isinstance(v, int):
            if v in {1,2,3,4}:
                return {1:"PR",2:"R",3:"P",4:"A"}[v]
        return None

    for key in ("gameType","seasonType","type","game_type"):
        v = item.get(key)
        tag = normalize(v)
        if tag:
            return tag
    meta = item.get("gameSchedule") or {}
    tag = normalize(meta.get("gameType"))
    return tag or "UNK"

def _maybe_tqdm(it: Iterable, enable: bool, desc: str):
    if enable:
        return tqdm(it, desc=desc, leave=False, file=sys.stderr)
    return it

def list_game_ids_for_season(season_start_year: int,
                             include_regular: bool = True,
                             include_playoffs: bool = True,
                             progress: bool = SHOW_PROGRESS) -> List[int]:
    season = season_str(season_start_year)
    ids: Set[int] = set()
    n_items = 0

    teams = tricodes_for_season(season_start_year)
    for tri in _maybe_tqdm(teams, progress, desc=f"{season} teams"):
        url = f"{API_BASE}/club-schedule-season/{tri}/{season}"
        data = get_json(url)
        games = data.get("games") if isinstance(data, dict) else data
        if not isinstance(games, list):
            games = []

        for g in games:
            n_items += 1
            tag = _extract_game_type(g)
            if (tag == "R" and include_regular) or (tag == "P" and include_playoffs):
                gid = _extract_game_id(g)
                if gid is not None:
                    ids.add(gid)
        time.sleep(REQUEST_PAUSE_SEC)

    out = sorted(ids)
    print(f"[{season_start_year}-{season_start_year+1}] scanned {len(teams)} teams,"
          f" {n_items} items -> {len(out)} games "
          f"({'R' if include_regular else ''}{'+' if include_regular and include_playoffs else ''}{'P' if include_playoffs else ''})")
    return out