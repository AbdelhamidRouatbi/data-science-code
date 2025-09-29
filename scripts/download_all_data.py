"""
NHL Data Downloader — general and playoff games
Downloads all regular season and playoff games for each year.
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm


class NHLDataDownloader:
    def __init__(self, base_dir="data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # OLD API endpoint (required by milestone)
        self.base_url = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live"
        
        # Keep using new API for schedule (it still works for getting game IDs)
        self.schedule_url = "https://api-web.nhle.com/v1/schedule/{}"

        # Season ranges: regular season + playoffs
        self.seasons = {
            "2016": ("2016-10-01", "2017-06-30"),
            "2017": ("2017-10-01", "2018-06-30"),
            "2018": ("2018-10-01", "2019-06-30"),
            "2019": ("2019-10-01", "2020-10-31"),
            "2020": ("2020-01-01", "2021-07-31"),
            "2021": ("2021-10-01", "2022-06-30"),
            "2022": ("2022-10-01", "2023-06-30"),
            "2023": ("2023-10-01", "2024-06-30"),
        }
    def get_folder(self, season, game_type):
        folder = self.base_dir / f"season_{season}" / game_type
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def get_game_ids(self, start_date, end_date, game_type_filter=None):
        """Get all game IDs between dates, optionally filtered by type."""
        print(f"Fetching schedule from {start_date} to {end_date}...")
        current_date = start_date
        game_ids = set()

        while current_date <= end_date:
            url = self.schedule_url.format(current_date)
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for game_day in data.get("gameWeek", []):
                        for game in game_day.get("games", []):
                            if game_type_filter is None or game.get("gameType") == game_type_filter:
                                game_ids.add(str(game.get("id")))
            except Exception as e:
                print(f"Error fetching schedule for {current_date}: {e}")

            current_date = self.increment_date(current_date)

        return sorted(list(game_ids))

    def increment_date(self, date_str):
        """Increment date string by one day."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt += timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    def download_game(self, game_id, folder):
        url = self.base_url.format(game_id)
        file_path = folder / f"{game_id}.json"

        if file_path.exists():
            return True

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
        except Exception:
            return False

        return False

    def download_games_for_season(self, season, start_date, end_date, game_type, game_type_filter):
        folder = self.get_folder(season, game_type)
        game_ids = self.get_game_ids(start_date, end_date, game_type_filter)

        print(f"Found {len(game_ids)} {game_type} games for season {season}.")

        for game_id in tqdm(game_ids, desc=f"Downloading {season} {game_type}"):
            self.download_game(game_id, folder)
            time.sleep(0.1)

    def download_all(self):
        for season, (start_date, end_date) in self.seasons.items():
            print(f"\nDownloading season {season}...")

            # Regular season games (gameType=2)
            self.download_games_for_season(season, start_date, end_date, "general", game_type_filter=2)

            # Playoff games (gameType=3)
            self.download_games_for_season(season, start_date, end_date, "playoff", game_type_filter=3)


if __name__ == "__main__":
    downloader = NHLDataDownloader(base_dir="data/raw")
    downloader.download_all()
    print("\nAll games downloaded.")
