"""
NHL Data Processor - Clean Version
Processes NHL game data with only essential columns for analysis.
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

class NHLDataProcessor:
    """
    Processes NHL game data from JSON to CSV format with essential columns only.
    """
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Team ID to name mapping
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
        
        # Player ID to name mapping
        self.player_mapping = {}
    
    def build_player_mapping(self):
        """Build player ID to name mapping from game roster data"""
        print("Building player name mapping...")
        
        for season in range(2016, 2024):
            season_dir = self.raw_data_dir / f"season_{season}" / "general"
            if season_dir.exists():
                json_files = list(season_dir.glob("*.json"))[:3]  # Sample 3 games per season
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            game_data = json.load(f)
                        
                        roster_spots = game_data.get('rosterSpots', [])
                        for player in roster_spots:
                            player_id = str(player.get('playerId', ''))
                            if player_id and player_id not in self.player_mapping:
                                first_name = player.get('firstName', {}).get('default', '')
                                last_name = player.get('lastName', {}).get('default', '')
                                self.player_mapping[player_id] = f"{first_name} {last_name}".strip()
                    
                    except Exception:
                        continue
        
        print(f"Built player mapping with {len(self.player_mapping)} players")
    
    def process_all_data(self):
        """Process all NHL data with essential columns only"""
        self.build_player_mapping()
        
        valid_seasons = [f"season_{year}" for year in range(2016, 2024)]
        season_folders = [f for f in self.raw_data_dir.iterdir() 
                         if f.is_dir() and f.name in valid_seasons]
        
        print(f"Processing {len(season_folders)} season folders")
        
        for season_folder in season_folders:
            season_name = season_folder.name
            print(f"Processing {season_name}")
            
            general_dir = season_folder / "general"
            if general_dir.exists():
                self._process_game_type_folder(general_dir, season_name, "general")
            
            playoff_dir = season_folder / "playoff"
            if playoff_dir.exists():
                self._process_game_type_folder(playoff_dir, season_name, "playoff")
    
    def _process_game_type_folder(self, raw_folder, season_name, game_type):
        """Process all JSON files in a folder"""
        json_files = list(raw_folder.glob("*.json"))
        print(f"  {game_type}: {len(json_files)} games found")
        
        if not json_files:
            return
        
        processed_folder = self.processed_data_dir / season_name / game_type
        processed_folder.mkdir(parents=True, exist_ok=True)
        
        all_events = []
        
        for json_file in tqdm(json_files, desc=f"    Processing {game_type}"):
            try:
                game_events = self._process_single_game(json_file, season_name, game_type)
                if game_events:
                    all_events.extend(game_events)
                    
                    game_id = json_file.stem
                    game_df = pd.DataFrame(game_events)
                    csv_path = processed_folder / f"{game_id}.csv"
                    game_df.to_csv(csv_path, index=False)
                    
            except Exception:
                continue
        
        if all_events:
            combined_df = pd.DataFrame(all_events)
            combined_df = self._calculate_additional_features(combined_df)
            
            combined_csv_path = processed_folder / f"all_{game_type}_games.csv"
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"Saved {len(combined_df)} events to {combined_csv_path}")
            
            # Show sample of the clean data
            sample_df = combined_df.head(3)
            print("Sample data:")
            print(sample_df[['game_id', 'event_type', 'team_name', 'player_name', 'goalie_name', 'shot_type']])
    
    def _process_single_game(self, json_file, season_name, game_type):
        """Extract events with clean column structure"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
        except Exception:
            return []
        
        game_player_mapping = self._extract_game_player_mapping(game_data)
        
        game_id = json_file.stem
        events = []
        plays = game_data.get('plays', [])
        
        for play in plays:
            event_type = play.get('typeDescKey', '')
            
            if event_type not in ['shot-on-goal', 'goal']:
                continue
            
            event_data = self._extract_event_data(play, game_id, season_name, game_type, event_type, game_player_mapping)
            if event_data:
                events.append(event_data)
        
        return events
    
    def _extract_game_player_mapping(self, game_data):
        """Extract player names from a specific game"""
        player_mapping = {}
        roster_spots = game_data.get('rosterSpots', [])
        
        for player in roster_spots:
            player_id = str(player.get('playerId', ''))
            if player_id:
                first_name = player.get('firstName', {}).get('default', '')
                last_name = player.get('lastName', {}).get('default', '')
                player_mapping[player_id] = f"{first_name} {last_name}".strip()
        
        return player_mapping
    
    def _extract_event_data(self, play, game_id, season_name, game_type, event_type, player_mapping):
        """Extract event data with clean columns only"""
        try:
            details = play.get('details', {})
            
            event_data = {
                'game_id': game_id,
                'season': season_name.replace('season_', ''),
                'game_type': game_type,
                'event_type': event_type.upper().replace('-', '_'),
                'period': play.get('periodDescriptor', {}).get('number', 1),
                'period_time': play.get('timeInPeriod', ''),
                'x_coord': details.get('xCoord'),
                'y_coord': details.get('yCoord'),
                'shot_type': details.get('shotType', ''),
                'team_id': details.get('eventOwnerTeamId', ''),
            }
            
            # Map team ID to team name
            team_id = str(event_data['team_id'])
            event_data['team_name'] = self.team_mapping.get(team_id, f'Unknown_{team_id}')
            
            # Handle player names - simplified approach
            if event_type == 'shot-on-goal':
                shooter_id = str(details.get('shootingPlayerId', ''))
                goalie_id = str(details.get('goalieInNetId', ''))
                
                event_data['player_name'] = player_mapping.get(shooter_id, 
                    self.player_mapping.get(shooter_id, f'Player_{shooter_id}'))
                event_data['goalie_name'] = player_mapping.get(goalie_id, 
                    self.player_mapping.get(goalie_id, f'Goalie_{goalie_id}'))
                    
            elif event_type == 'goal':
                scorer_id = str(details.get('scoringPlayerId', ''))
                goalie_id = str(details.get('goalieInNetId', ''))
                
                event_data['player_name'] = player_mapping.get(scorer_id, 
                    self.player_mapping.get(scorer_id, f'Player_{scorer_id}'))
                event_data['goalie_name'] = player_mapping.get(goalie_id, 
                    self.player_mapping.get(goalie_id, f'Goalie_{goalie_id}'))
            
            return event_data
            
        except Exception:
            return None
    
    def _calculate_additional_features(self, df):
        """Calculate analytical features"""
        df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
        df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
        
        valid_coords = df['x_coord'].notna() & df['y_coord'].notna()
        df.loc[valid_coords, 'distance_from_net'] = (
            (df.loc[valid_coords, 'x_coord'] - 89) ** 2 + 
            df.loc[valid_coords, 'y_coord'] ** 2
        ) ** 0.5
        
        df['is_goal'] = (df['event_type'] == 'GOAL').astype(int)
        
        return df

def main():
    processor = NHLDataProcessor("data/raw", "data/processed")
    processor.process_all_data()
    print("Data processing complete.")

if __name__ == "__main__":
    main()