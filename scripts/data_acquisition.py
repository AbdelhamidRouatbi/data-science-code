import requests
import json
import time
import os
from pathlib import Path
import pandas as pd

class NHLDownloader:
    """
    A simple class to get NHL play-by-play data using the NHL API.
    It downloads both regular season and playoff games from 2016 to 2023.
    """

    def __init__(self, save_dir="data/raw"):
        # make the folder to save files
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # API link
        self.base_url = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

        # seasons dictionary
        self.seasons = {
            "2016": "20162017",
            "2017": "20172018",
            "2018": "20182019",
            "2019": "20192020",
            "2020": "20202021",
            "2021": "20212022",
            "2022": "20222023",
            "2023": "20232024"
        }

    def make_game_ids(self, season, game_type="02"):
        """
        Creates a list of game ids for the given season.
        game_type "02" = regular season, "03" = playoffs
        """
        ids = []
        season_code = self.seasons[season]

        if game_type == "02":
            # max ~1311 games in regular season
            for i in range(1, 1312):
                ids.append(f"{season_code}{game_type}{i:04d}")
        else:
            # playoffs, round/matchup/game
            for r in range(1, 5):
                for m in range(1, 9):
                    for g in range(1, 8):
                        ids.append(f"{season_code}{game_type}{r}{m}{g}")
        return ids

    def download_one_game(self, game_id, redownload=False):
        """
        Downloads one game by its id.
        """
        file_path = self.save_dir / f"{game_id}.json"

        # if file already exists and we don’t want to redownload
        if file_path.exists() and not redownload:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass  # if file is broken, we try again

        url = self.base_url.format(game_id=game_id)
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if "id" in data and "plays" in data:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    return data
        except:
            return None
        return None

    def download_season(self, season, game_type="02", limit=None, delay=0.2):
        """
        Downloads all games for one season.
        """
        ids = self.make_game_ids(season, game_type)
        if limit:
            ids = ids[:limit]

        name = "regular season" if game_type == "02" else "playoffs"
        print(f"Downloading {len(ids)} {name} games for {season}...")

        downloaded = {}
        for gid in ids:
            data = self.download_one_game(gid)
            if data:
                downloaded[gid] = data
            time.sleep(delay)  # don’t overload the server
        print(f"{season} {name} done: {len(downloaded)} files")
        return downloaded

    def download_everything(self, seasons=None, max_games=None):
        """
        Downloads both regular season and playoff data for all seasons.
        """
        if seasons is None:
            seasons = list(self.seasons.keys())

        all_data = {}
        for s in seasons:
            print("="*40)
            print("Season:", s)
            print("="*40)

            all_data.update(self.download_season(s, "02", max_games))
            all_data.update(self.download_season(s, "03", max_games))

        print("All done. Total games:", len(all_data))
        return all_data

    def stats(self):
        """
        Makes a quick dataframe with info about downloaded files.
        """
        files = [f for f in self.save_dir.glob("*.json")]
        rows = []
        for f in files:
            game_id = f.stem
            season = game_id[:4]
            gtype = "Regular" if game_id[4:6] == "02" else "Playoff"
            try:
                with open(f, "r", encoding="utf-8") as j:
                    d = json.load(j)
                nplays = len(d.get("plays", []))
            except:
                nplays = 0
            rows.append({
                "game_id": game_id,
                "season": season,
                "type": gtype,
                "size_MB": f.stat().st_size / (1024*1024),
                "plays": nplays
            })
        return pd.DataFrame(rows)


def download_nhl(seasons=None, max_games=None, save_dir="data/raw"):
    """
    Shortcut to start a download.
    """
    dl = NHLDownloader(save_dir)
    return dl.download_everything(seasons, max_games)
