"""
Team tricodes and helpers that depend on the season.
"""

from __future__ import annotations
from typing import List

ALL_TRICODES = [
    "ANA","ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM","FLA","LAK",
    "MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SJS","STL","TBL","TOR","VAN",
    "WPG","WSH","VGK","SEA"
]

def season_str(season_start_year: int) -> str:
    """2016 -> '20162017'"""
    return f"{season_start_year}{season_start_year+1}"

def tricodes_for_season(season_start_year: int) -> List[str]:
    """Return teams active that season (VGK from 2017-18, SEA from 2021-22)."""
    have = [t for t in ALL_TRICODES if t not in {"VGK","SEA"}]
    if season_start_year >= 2017:
        have.append("VGK")
    if season_start_year >= 2021:
        have.append("SEA")
    return sorted(have)