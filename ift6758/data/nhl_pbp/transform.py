# nhl_pbp/transform.py
from __future__ import annotations
import csv, json, os
from typing import Dict, Iterable, List, Optional

EVENT_COLUMNS = [
    "game_id","season","game_type","event_type","period","period_time",
    "x_coord","y_coord","shot_type","team_id","team_name","player_name","goalie_name",
]

# --- add near the top, with other helpers ---
def _map_game_type_code(code) -> str:
    # Accept numeric, "01"/"02" strings, or letter-y variants
    if code in (1, "1", "01", "PR", "PRESEASON"): return "preseason"
    if code in (2, "2", "02", "R", "REGULAR"):   return "regular"
    if code in (3, "3", "03", "P", "PLAYOFFS"):  return "playoffs"
    if code in (4, "4", "04", "A", "ALL-STAR", "ALLSTAR", "ASG"): return "all-star"
    return "unknown"

def _derive_game_type(data: dict) -> str:
    gt = data.get("gameType")
    # Try JSON field first
    if gt is not None:
        mt = _map_game_type_code(gt if not isinstance(gt, str) else gt.upper())
        if mt != "unknown":
            return mt
    # Fallback: parse from game id (YYYYSSXXXX -> SS are digits 5-6)
    gid = data.get("id")
    if gid is not None:
        s = str(gid)
        if len(s) >= 6:
            mt = _map_game_type_code(s[4:6])
            if mt != "unknown":
                return mt
    return "unknown"

def _season_start_year(season_field: Optional[int]) -> Optional[int]:
    # 20192020 -> 2019
    if season_field:
        s = str(season_field)
        if len(s) >= 4:
            return int(s[:4])
    return None

def _season_from_id(gid) -> Optional[int]:
    if gid is None:
        return None
    s = str(gid)
    return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None


def _name_map(roster_spots: List[dict]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    for rs in roster_spots or []:
        pid = rs.get("playerId")
        fn = (rs.get("firstName") or {}).get("default") or ""
        ln = (rs.get("lastName") or {}).get("default") or ""
        if pid is not None:
            m[pid] = f"{fn} {ln}".strip() or None
    return m

def _team_map(home: dict, away: dict) -> Dict[int, str]:
    def full_name(t: dict) -> str:
        place = (t.get("placeName") or {}).get("default") or ""
        common = (t.get("commonName") or {}).get("default") or ""
        return f"{place} {common}".strip() or None
    tm: Dict[int, str] = {}
    if home and "id" in home: tm[home["id"]] = full_name(home)
    if away and "id" in away: tm[away["id"]] = full_name(away)
    return tm


def _iter_rows_from_game_json(data: dict) -> Iterable[List[object]]:
    names = _name_map(data.get("rosterSpots", []))
    teams = _team_map(data.get("homeTeam", {}), data.get("awayTeam", {}))
    gid = data.get("id")

    season = _season_start_year(data.get("season")) or _season_from_id(gid)
    game_type_str = _derive_game_type(data)  # <-- NEW

    for play in data.get("plays", []) or []:
        type_key = (play.get("typeDescKey") or "").lower()
        if type_key not in ("goal", "shot-on-goal"):
            continue

        det = play.get("details", {}) or {}
        event_type = "GOAL" if type_key == "goal" else "SHOT_ON_GOAL"
        shooter_id = det.get("shootingPlayerId") or det.get("scoringPlayerId")
        goalie_id = det.get("goalieInNetId")
        team_id = det.get("eventOwnerTeamId")

        row = [
            gid,
            season,
            game_type_str,  # <-- was "general"
            event_type,
            (play.get("periodDescriptor") or {}).get("number"),
            play.get("timeInPeriod"),
            det.get("xCoord"),
            det.get("yCoord"),
            (det.get("shotType") or None) and det.get("shotType").lower(),
            team_id,
            teams.get(team_id),
            names.get(shooter_id),
            names.get(goalie_id),
        ]
        yield row

def json_to_csv(json_path: str, out_csv_path: str) -> int:
    """
    Convert one game JSON to a CSV matching EVENT_COLUMNS.
    Returns number of rows written.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = list(_iter_rows_from_game_json(data))
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EVENT_COLUMNS)
        w.writerows(rows)
    return len(rows)

def season_jsons_to_csvs_via_cache(
    season_start_year: int,
    out_dir: str,
    merged_out_path: Optional[str] = None,
) -> int:
    """
    Iterate cached game JSONs for a season (using cache.iter_cached_games),
    write per-game CSVs to out_dir, and optionally a merged CSV.
    Returns total rows written (sum across games).
    """
    from .cache import iter_cached_games  # you already have this

    os.makedirs(out_dir, exist_ok=True)

    # Setup merged writer if requested (streamed)
    merged_f = None
    merged_w = None
    total = 0

    try:
        if merged_out_path:
            os.makedirs(os.path.dirname(merged_out_path) or ".", exist_ok=True)
            merged_f = open(merged_out_path, "w", newline="", encoding="utf-8")
            merged_w = csv.writer(merged_f)
            merged_w.writerow(EVENT_COLUMNS)

        for game_id, json_path in iter_cached_games(season_start_year):
            # per-game
            out_csv = os.path.join(out_dir, f"{game_id}.csv")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = list(_iter_rows_from_game_json(data))

            # write per-game
            with open(out_csv, "w", newline="", encoding="utf-8") as g:
                w = csv.writer(g)
                w.writerow(EVENT_COLUMNS)
                w.writerows(rows)

            # stream-append to merged
            if merged_w:
                merged_w.writerows(rows)

            total += len(rows)

        return total
    finally:
        if merged_f:
            merged_f.close()