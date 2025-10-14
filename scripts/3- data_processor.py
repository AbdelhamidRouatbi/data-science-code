"""
NHL Data Processor - BASIC VERSION
Extracts raw game data and saves it to CSV
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

class NHLDataProcessor:
    """Reads NHL game data and saves basic info to CSV"""

    def __init__(self, raw_data_dir=None, processed_data_dir=None):
        # Set folders for input and output data
        if raw_data_dir is None:
            self.raw_data_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\raw")
        else:
            self.raw_data_dir = Path(raw_data_dir)

        if processed_data_dir is None:
            self.processed_data_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\processed")
        else:
            self.processed_data_dir = Path(processed_data_dir)

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Map team IDs to team names
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

    def process_all_data(self):
        """Goes through all seasons and processes their games"""
        season_folders = [f for f in self.raw_data_dir.iterdir() if f.is_dir() and f.name.startswith('season_')]
        
        for season_folder in season_folders:
            season_name = season_folder.name
            print(f"Processing {season_name}")
            
            for folder_name in ["general", "playoff"]:
                folder = season_folder / folder_name
                if folder.exists():
                    self._process_game_type_folder(folder, season_name, folder_name)

    def _process_game_type_folder(self, raw_folder, season_name, game_type):
        """Processes all games of one type (regular or playoffs)"""
        json_files = list(raw_folder.glob("*.json"))
        print(f"  {game_type}: {len(json_files)} games found")
        
        if not json_files:
            return
            
        processed_folder = self.processed_data_dir / season_name / game_type
        processed_folder.mkdir(parents=True, exist_ok=True)
        all_events = []

        for json_file in tqdm(json_files, desc=f"Processing {game_type}"):
            try:
                game_events = self._process_single_game(json_file, season_name, game_type)
                if game_events:
                    all_events.extend(game_events)
                    game_df = pd.DataFrame(game_events)
                    game_df.to_csv(processed_folder / f"{json_file.stem}.csv", index=False)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        if all_events:
            combined_df = pd.DataFrame(all_events)
            combined_df.to_csv(processed_folder / f"all_{game_type}_games.csv", index=False)
            print(f"Saved {len(combined_df)} events for {game_type}")

    def _process_single_game(self, json_file, season_name, game_type):
        """Reads one game and extracts shot and goal events"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return []

        game_id = json_file.stem
        events = []
        plays = game_data.get('plays', [])
        
        for play in plays:
            event_type = play.get('typeDescKey', '')
            if event_type not in ['shot-on-goal', 'goal']:
                continue
            
            event = self._extract_event_data(play, game_id, season_name, game_type, event_type)
            if event:
                events.append(event)
        
        return events

    def _extract_event_data(self, play, game_id, season_name, game_type, event_type):
        """Extracts info from one play"""
        try:
            details = play.get('details', {})
            period_info = play.get('periodDescriptor', {})
            
            event = {
                'game_id': game_id,
                'season': season_name.replace('season_', ''),
                'game_type': game_type,
                'event_type': event_type.upper().replace('-', '_'),
                'period': period_info.get('number', 1),
                'period_time': play.get('timeInPeriod', ''),
                'x_coord': details.get('xCoord'),
                'y_coord': details.get('yCoord'),
                'shot_type': details.get('shotType', ''),
                'team_id': str(details.get('eventOwnerTeamId', '')),
            }
            
            team_id = str(event['team_id'])
            event['team_name'] = self.team_mapping.get(team_id, f'Unknown_{team_id}')

            if event_type == 'shot-on-goal':
                shooter_id = str(details.get('shootingPlayerId', ''))
                goalie_id = str(details.get('goalieInNetId', ''))
            else:
                shooter_id = str(details.get('scoringPlayerId', ''))
                goalie_id = str(details.get('goalieInNetId', ''))

            event['player_name'] = f"Player_{shooter_id}" if shooter_id else ''
            event['goalie_name'] = f"Goalie_{goalie_id}" if goalie_id else ''

            if event_type == 'goal':
                event['is_goal'] = 1
                event['empty_net'] = details.get('emptyNet', False)
                event['strength'] = details.get('strength', 'EVEN')
                event['game_winning_goal'] = details.get('gameWinningGoal', False)
            else:
                event['is_goal'] = 0
                event['empty_net'] = False
                event['strength'] = ''
                event['game_winning_goal'] = False

            return event
            
        except Exception as ex:
            print(f"Error extracting event data: {ex}")
            return None

def main():
    processor = NHLDataProcessor(
        raw_data_dir=r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\raw",
        processed_data_dir=r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\processed"
    )
    processor.process_all_data()
    print("Basic data processing complete.")

if __name__ == "__main__":
    main()
