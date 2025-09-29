import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

class NHLDataProcessor:
    """
    Simple NHL Data Processor
    - Reads raw game data (JSON)
    - Extracts only shots and goals
    - Saves them into tidy CSV files
    """
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Map team IDs to their names (from NHL API)
        self.team_mapping = {
            '1': 'New Jersey Devils', '2': 'New York Islanders', '3': 'New York Rangers',
            '4': 'Philadelphia Flyers', '5': 'Pittsburgh Penguins', '6': 'Boston Bruins',
            '7': 'Buffalo Sabres', '8': 'Montréal Canadiens', '9': 'Ottawa Senators',
            '10': 'Toronto Maple Leafs', '12': 'Carolina Hurricanes', '13': 'Florida Panthers',
            '14': 'Tampa Bay Lightning', '15': 'Washington Capitals', '16': 'Chicago Blackhawks',
            '17': 'Detroit Red Wings', '18': 'Nashville Predators', '19': 'St. Louis Blues',
            '20': 'Calgary Flames', '21': 'Colorado Avalanche', '22': 'Edmonton Oilers',
            '23': 'Vancouver Canucks', '24': 'Anaheim Ducks', '25': 'Dallas Stars',
            '26': 'Los Angeles Kings', '28': 'San Jose Sharks', '29': 'Columbus Blue Jackets',
            '30': 'Minnesota Wild', '52': 'Winnipeg Jets', '53': 'Arizona Coyotes',
            '54': 'Vegas Golden Knights', '55': 'Seattle Kraken'
        }
        
        # Player IDs will be stored here later
        self.player_mapping = {}
    
    def build_player_mapping(self):
        """Build a dictionary that maps player IDs to names from the roster info"""
        print("Building player name mapping...")
        
        for season in range(2016, 2024):
            season_dir = self.raw_data_dir / f"season_{season}" / "general"
            if not season_dir.exists():
                continue
            
            # Only look at a few games to save time
            json_files = list(season_dir.glob("*.json"))[:3]
            
            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        game_data = json.load(f)
                    
                    for player in game_data.get("rosterSpots", []):
                        player_id = str(player.get("playerId", ""))
                        if player_id and player_id not in self.player_mapping:
                            first_name = player.get("firstName", {}).get("default", "")
                            last_name = player.get("lastName", {}).get("default", "")
                            self.player_mapping[player_id] = f"{first_name} {last_name}".strip()
                except:
                    continue
        
        print(f"Collected {len(self.player_mapping)} player names")
    
    def process_all_data(self):
        """Go through all seasons and process games"""
        self.build_player_mapping()
        
        valid_seasons = [f"season_{year}" for year in range(2016, 2024)]
        season_folders = [f for f in self.raw_data_dir.iterdir() 
                         if f.is_dir() and f.name in valid_seasons]
        
        print(f"Found {len(season_folders)} season folders")
        
        for season_folder in season_folders:
            season_name = season_folder.name
            print(f"Processing {season_name}...")
            
            for game_type in ["general", "playoff"]:
                raw_folder = season_folder / game_type
                if raw_folder.exists():
                    self._process_game_type_folder(raw_folder, season_name, game_type)
    
    def _process_game_type_folder(self, raw_folder, season_name, game_type):
        """Process all games inside one type (general or playoff)"""
        json_files = list(raw_folder.glob("*.json"))
        print(f"  {game_type}: {len(json_files)} games")
        
        if not json_files:
            return
        
        processed_folder = self.processed_data_dir / season_name / game_type
        processed_folder.mkdir(parents=True, exist_ok=True)
        
        all_events = []
        
        for json_file in tqdm(json_files, desc=f"    {game_type}"):
            try:
                game_events = self._process_single_game(json_file, season_name, game_type)
                if game_events:
                    all_events.extend(game_events)
                    
                    # Save individual game
                    game_id = json_file.stem
                    pd.DataFrame(game_events).to_csv(processed_folder / f"{game_id}.csv", index=False)
            except:
                continue
        
        # Save all games together
        if all_events:
            combined_df = pd.DataFrame(all_events)
            combined_df = self._calculate_additional_features(combined_df)
            combined_df.to_csv(processed_folder / f"all_{game_type}_games.csv", index=False)
            print(f"    Saved {len(combined_df)} events for {game_type}")
    
    def _process_single_game(self, json_file, season_name, game_type):
        """Get events (shots + goals) from one game"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                game_data = json.load(f)
        except:
            return []
        
        player_mapping = self._extract_game_player_mapping(game_data)
        game_id = json_file.stem
        events = []
        
        for play in game_data.get("plays", []):
            event_type = play.get("typeDescKey", "")
            if event_type not in ["shot-on-goal", "goal"]:
                continue
            
            event = self._extract_event_data(play, game_id, season_name, game_type, event_type, player_mapping)
            if event:
                events.append(event)
        
        return events
    
    def _extract_game_player_mapping(self, game_data):
        """Extract player names for this specific game"""
        mapping = {}
        for player in game_data.get("rosterSpots", []):
            player_id = str(player.get("playerId", ""))
            if player_id:
                first_name = player.get("firstName", {}).get("default", "")
                last_name = player.get("lastName", {}).get("default", "")
                mapping[player_id] = f"{first_name} {last_name}".strip()
        return mapping
    
    def _extract_event_data(self, play, game_id, season_name, game_type, event_type, player_mapping):
        """Pick out the important details from an event"""
        try:
            details = play.get("details", {})
            data = {
                "game_id": game_id,
                "season": season_name.replace("season_", ""),
                "game_type": game_type,
                "event_type": event_type.upper().replace("-", "_"),
                "period": play.get("periodDescriptor", {}).get("number", 1),
                "period_time": play.get("timeInPeriod", ""),
                "x_coord": details.get("xCoord"),
                "y_coord": details.get("yCoord"),
                "shot_type": details.get("shotType", ""),
                "team_id": details.get("eventOwnerTeamId", ""),
            }
            
            # Team name
            data["team_name"] = self.team_mapping.get(str(data["team_id"]), "Unknown")
            
            # Shooter / Scorer and Goalie
            if event_type == "shot-on-goal":
                shooter_id = str(details.get("shootingPlayerId", ""))
                goalie_id = str(details.get("goalieInNetId", ""))
                data["player_name"] = player_mapping.get(shooter_id, f"Player_{shooter_id}")
                data["goalie_name"] = player_mapping.get(goalie_id, f"Goalie_{goalie_id}")
            
            elif event_type == "goal":
                scorer_id = str(details.get("scoringPlayerId", ""))
                goalie_id = str(details.get("goalieInNetId", ""))
                data["player_name"] = player_mapping.get(scorer_id, f"Player_{scorer_id}")
                data["goalie_name"] = player_mapping.get(goalie_id, f"Goalie_{goalie_id}")
            
            return data
        except:
            return None
    
    def _calculate_additional_features(self, df):
        """Add extra stats like distance from net and if it was a goal"""
        df["x_coord"] = pd.to_numeric(df["x_coord"], errors="coerce")
        df["y_coord"] = pd.to_numeric(df["y_coord"], errors="coerce")
        
        valid = df["x_coord"].notna() & df["y_coord"].notna()
        df.loc[valid, "distance_from_net"] = (
            (df.loc[valid, "x_coord"] - 89) ** 2 + df.loc[valid, "y_coord"] ** 2
        ) ** 0.5
        
        df["is_goal"] = (df["event_type"] == "GOAL").astype(int)
        
        return df

def main():
    processor = NHLDataProcessor("data/raw", "data/processed")
    processor.process_all_data()
    print("Done!")

if __name__ == "__main__":
    main()
