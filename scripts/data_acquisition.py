import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm


class NHLDataAcquisition:
    """
    Download and manage NHL play-by-play data from the new NHL API.
    Covers regular season and playoff games from 2016–2023.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # API endpoint
        self.base_url = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

        # Map short season keys to full codes
        self.seasons = {
            '2016': '20162017',
            '2017': '20172018',
            '2018': '20182019',
            '2019': '20192020',
            '2020': '20202021',
            '2021': '20212022',
            '2022': '20222023',
            '2023': '20232024'
        }

    def generate_game_ids(self, season: str, game_type: str = "02") -> List[str]:
        """
        Generate game IDs for a season.

        season: e.g. '2016'
        game_type: '02' = regular season, '03' = playoffs
        """
        season_code = self.seasons[season]
        game_ids = []

        if game_type == "02":
            # Regular season: up to 1311 games
            for game_number in range(1, 1312):
                game_ids.append(f"{season_code}{game_type}{game_number:04d}")
        else:
            # Playoffs: rounds, matchups, and games
            for round_num in range(1, 5):
                for matchup in range(1, 9):
                    for game_num in range(1, 8):
                        game_ids.append(f"{season_code}{game_type}{round_num}{matchup}{game_num}")

        return game_ids

    def download_game_data(self, game_id: str, force_redownload: bool = False) -> Optional[Dict]:
        """
        Download data for a single game. Returns a dict or None.
        """
        file_path = self.data_dir / f"{game_id}.json"

        if file_path.exists() and not force_redownload:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass  # if file is corrupted, redownload

        url = self.base_url.format(game_id=game_id)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "id" in data and "plays" in data:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return data
            return None
        except Exception:
            return None

    def download_season_data(self, season: str, game_type: str = "02",
                             max_games: int = None, delay: float = 0.2) -> Dict[str, Dict]:
        """
        Download all games for one season (regular or playoffs).
        """
        game_ids = self.generate_game_ids(season, game_type)
        if max_games:
            game_ids = game_ids[:max_games]

        game_type_name = "regular season" if game_type == "02" else "playoffs"
        print(f"Downloading {len(game_ids)} {game_type_name} games for {season}...")

        downloaded = {}
        for game_id in tqdm(game_ids, desc=f"{season} {game_type_name}"):
            data = self.download_game_data(game_id)
            if data:
                downloaded[game_id] = data
            time.sleep(delay)

        print(f"Finished {season} {game_type_name}: {len(downloaded)} games saved")
        return downloaded

    def download_all_data(self, seasons: List[str] = None, max_games_per_season: int = None):
        """
        Download data for all selected seasons.
        """
        if seasons is None:
            seasons = list(self.seasons.keys())

        all_data = {}
        for season in seasons:
            print("=" * 60)
            print("Processing season:", season)
            print("=" * 60)

            # Regular season
            all_data.update(self.download_season_data(season, "02", max_games_per_season))
            # Playoffs
            all_data.update(self.download_season_data(season, "03", max_games_per_season))

        print("Download complete. Total games:", len(all_data))
        self._save_download_summary(all_data, seasons)
        return all_data

    def _save_download_summary(self, all_data: Dict, seasons: List[str]):
        """Save a summary file of what was downloaded."""
        summary = {
            "total_games": len(all_data),
            "seasons_downloaded": seasons,
            "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "games_per_season": {}
        }
        for game_id in all_data.keys():
            season = game_id[:4]
            summary["games_per_season"][season] = summary["games_per_season"].get(season, 0) + 1

        summary_file = self.data_dir / "download_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print("Summary saved to:", summary_file)

    def get_download_stats(self) -> pd.DataFrame:
        """
        Return statistics about downloaded data as a DataFrame.
        """
        files = [f for f in self.data_dir.glob("*.json") if f.name != "download_summary.json"]

        stats = []
        for file_path in files:
            game_id = file_path.stem
            season = game_id[:4]
            game_type = "Regular" if game_id[4:6] == "02" else "Playoff"

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                num_plays = len(data.get("plays", []))
            except Exception:
                num_plays = 0

            stats.append({
                "game_id": game_id,
                "season": season,
                "game_type": game_type,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "num_plays": num_plays
            })

        return pd.DataFrame(stats)


def download_nhl_data(seasons=None, max_games=None, data_dir="data/raw"):
    """
    Simple function to start a full NHL data download.
    """
    acquirer = NHLDataAcquisition(data_dir)
    acquirer.download_all_data(seasons, max_games)
    return acquirer
