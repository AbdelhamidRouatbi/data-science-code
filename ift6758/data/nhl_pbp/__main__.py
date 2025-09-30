"""
Command line interface with progress bar support (tqdm).

Examples:
  python -m nhl_pbp ids 2016
  python -m nhl_pbp season 2016 --limit 50
  python -m nhl_pbp fetch 2017020001 2017020002 --force
  python -m nhl_pbp seasons --start 2016 --end 2023
  python -m nhl_pbp manifest 2016 --out data/2016-2017_manifest.csv
"""

from __future__ import annotations
import argparse, sys
from typing import List
from .downloader import NHLPBPDownloader
from .config import REQUIRED_SEASONS, SHOW_PROGRESS
from .transform import json_to_csv, season_jsons_to_csvs_via_cache

def _add_common_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--regular", action="store_true", help="Include regular season games")
    p.add_argument("--playoffs", action="store_true", help="Include playoff games")
    p.add_argument("--limit", type=int, default=None, help="Limit number of games (for quick tests)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")

def _resolve_filters(args) -> tuple[bool,bool]:
    # Default to both if neither specified; else honor the flags
    inc_r = args.regular or (not args.regular and not args.playoffs)
    inc_p = args.playoffs or (not args.regular and not args.playoffs)
    return inc_r, inc_p

def _progress_from_args(args) -> bool:
    return False if args.no_progress else SHOW_PROGRESS

def cmd_ids(args) -> int:
    dl = NHLPBPDownloader()
    inc_r, inc_p = _resolve_filters(args)
    ids = dl.list_game_ids_for_season(args.season, include_regular=inc_r, include_playoffs=inc_p, progress=_progress_from_args(args))
    for gid in ids:
        print(gid)
    print(f"Total: {len(ids)}")
    return 0

def cmd_fetch(args) -> int:
    dl = NHLPBPDownloader()
    for gid in args.game_ids:
        dl.fetch_and_cache_pbp(int(gid), force=args.force)
        print(f"cached {gid}")
    return 0

def cmd_season(args) -> int:
    dl = NHLPBPDownloader()
    inc_r, inc_p = _resolve_filters(args)
    ids = dl.download_season(args.season, include_regular=inc_r, include_playoffs=inc_p,
                             limit=args.limit, progress=_progress_from_args(args))
    print(f"Done: {len(ids)} game ids processed (may be limited).")
    return 0

def cmd_seasons(args) -> int:
    dl = NHLPBPDownloader()
    start, end = args.start, args.end
    for y in range(start, end+1):
        inc_r, inc_p = _resolve_filters(args)
        dl.download_season(y, include_regular=inc_r, include_playoffs=inc_p,
                           limit=args.limit, progress=_progress_from_args(args))
    print("All seasons done.")
    return 0

def _season_dir(y: int) -> str:
    return f"{y}-{y+1}"

def cmd_pipeline(args) -> int:
    dl = NHLPBPDownloader()
    inc_r, inc_p = _resolve_filters(args)
    grand_total = 0

    for y in range(args.start, args.end + 1):
        # 1) Download (respects cache; won't re-download unless you add --force in fetch calls)
        dl.download_season(y, include_regular=inc_r, include_playoffs=inc_p,
                           limit=args.limit, progress=_progress_from_args(args))

        # 2) Convert from cache
        out_dir = args.out_dir_base.rstrip("/") + "/" + _season_dir(y)
        merged_out = (
            args.merged_base.rstrip("/") + "/" + _season_dir(y) + "_events.csv"
            if args.merged_base else None
        )
        total = season_jsons_to_csvs_via_cache(y, out_dir, merged_out)
        grand_total += total
        print(f"[{y}] downloaded+converted → {out_dir} (merged: {merged_out or 'none'})")

    print(f"Pipeline complete. Total rows across seasons: {grand_total}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nhl-pbp", description="NHL Play-by-Play Downloader (new NHL API)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ids", help="List game IDs for a season")
    sp.add_argument("season", type=int, help="Season start year (e.g., 2016 for 2016-17)")
    _add_common_filters(sp)
    sp.set_defaults(func=cmd_ids)

    sp = sub.add_parser("fetch", help="Fetch specific game ids and cache PBP")
    sp.add_argument("game_ids", nargs="+", help="One or more game ids")
    sp.add_argument("--force", action="store_true", help="Re-download even if cached")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("season", help="Download one whole season (regular + playoffs by default)")
    sp.add_argument("season", type=int)
    _add_common_filters(sp)
    sp.set_defaults(func=cmd_season)

    sp = sub.add_parser("seasons", help="Download a range of seasons (inclusive)")
    sp.add_argument("--start", type=int, default=REQUIRED_SEASONS[0], help="Start season (e.g., 2016)")
    sp.add_argument("--end", type=int, default=REQUIRED_SEASONS[-1], help="End season (e.g., 2023)")
    _add_common_filters(sp)
    sp.set_defaults(func=cmd_seasons)

    # sp = sub.add_parser("manifest", help="Write a manifest CSV for a season")
    # sp.add_argument("season", type=int)
    # sp.add_argument("--out", required=True, help="Output CSV path")
    # sp.set_defaults(func=cmd_manifest)
    
    # sp = sub.add_parser("to-csv", help="Convert a single game JSON to a CSV (GOAL + SHOT_ON_GOAL only)")
    # sp.add_argument("--json", required=True, help="Path to a game play-by-play JSON")
    # sp.add_argument("--out", required=True, help="Output CSV path")
    # sp.set_defaults(func=cmd_to_csv)

    # sp = sub.add_parser("to-csv-season", help="Convert all cached JSONs for a season to per-game CSVs and an optional merged CSV")
    # sp.add_argument("season", type=int, help="Season start year (e.g., 2016)")
    # sp.add_argument("--out-dir", required=True, help="Directory to write per-game CSVs")
    # sp.add_argument("--merged-out", help="(Optional) path for merged season CSV")
    # sp.set_defaults(func=cmd_to_csv_season)
    
    sp = sub.add_parser("pipeline", help="Download and convert a range of seasons in one go")
    sp.add_argument("--start", type=int, default=REQUIRED_SEASONS[0], help="Start season (e.g., 2016)")
    sp.add_argument("--end", type=int, default=REQUIRED_SEASONS[-1], help="End season (e.g., 2023)")
    _add_common_filters(sp)  # gives you --regular/--playoffs/--limit/--no-progress
    sp.add_argument("--out-dir-base", required=True, help="Base folder for per-game CSVs per season")
    sp.add_argument("--merged-base", help="Base folder for merged per-season CSVs; omit to skip merged")
    sp.set_defaults(func=cmd_pipeline)

    return p

def main(argv: List[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())