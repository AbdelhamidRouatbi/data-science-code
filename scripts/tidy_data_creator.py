import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

class NHLTidyDataCreator:
    """
    This class takes the messy raw NHL JSON files and turns them into a nice tidy dataframe.
    It also saves the results as CSV files, both for single games and combined seasons.
    """
    
    def __init__(self, raw_data_dir="data/raw", output_dir="data/tidy"):
        # where the raw JSON files are
        self.raw_data_dir = Path(raw_data_dir)
        # where we want to save the tidy CSV files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # map team IDs to team names (from the NHL API)
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
    
    def create_tidy_dataframe(self, seasons=None):
        """
        Go through all the JSON files and build one big tidy dataframe.
        We also save per-game CSVs and per-season CSVs.
        """
        if seasons is None:
            # default: seasons 2016–2023
            seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        
        all_seasons_data = []
        
        for season in seasons:
            print(f"Working on season {season}...")
            
            for game_type in ['general', 'playoff']:
                season_dir = self.raw_data_dir / f"season_{season}" / game_type
                
                if not season_dir.exists():
                    print(f"  Skipping {game_type}, folder not found")
                    continue
                
                json_files = list(season_dir.glob("*.json"))
                print(f"  Found {len(json_files)} {game_type} games")
                
                season_events = []
                
                for json_file in tqdm(json_files, desc=f"    {game_type}"):
                    try:
                        game_events = self._process_single_game(json_file, season, game_type)
                        season_events.extend(game_events)
                        
                        # save CSV for just this game
                        self._save_individual_game_csv(game_events, season, game_type, json_file.stem)
                    except Exception as e:
                        print(f"Problem with {json_file}: {e}")
                        continue
                
                # save all games for this season + game type
                if season_events:
                    self._save_season_game_type_csv(season_events, season, game_type)
                    all_seasons_data.extend(season_events)
        
        # save everything combined into one CSV
        if all_seasons_data:
            df = pd.DataFrame(all_seasons_data)
            df = self._calculate_additional_features(df)
            
            # keep only useful columns
            columns_to_keep = [
                'game_id', 'season', 'game_type', 'game_date', 'period', 'period_type', 
                'period_time', 'time_remaining', 'event_type', 'team_id', 'team_name', 
                'x_coord', 'y_coord', 'zone_code', 'shot_type', 'shooter_name', 'goalie_name',
                'empty_net', 'strength', 'game_winning_goal', 'scorer_name', 'assists',
                'distance_from_net', 'is_goal', 'shot_angle', 'period_time_seconds', 
                'game_time_seconds'
            ]
            
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            combined_path = self.output_dir / "all_seasons_combined.csv"
            df.to_csv(combined_path, index=False)
            print(f"Everything saved in {combined_path}")
            print(f"Total number of events: {len(df)}")
            
            # small sample for checking
            sample_path = self.output_dir / "tidy_data_sample.csv"
            df.head(100).to_csv(sample_path, index=False)
            print(f"Saved a sample in {sample_path}")
        
        return df
    
    def _save_individual_game_csv(self, game_events, season, game_type, game_id):
        """Save one game as a CSV"""
        if not game_events:
            return
        
        game_dir = self.output_dir / f"season_{season}" / game_type
        game_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(game_events)
        df = self._calculate_additional_features(df)
        
        game_path = game_dir / f"{game_id}.csv"
        df.to_csv(game_path, index=False)
    
    def _save_season_game_type_csv(self, season_events, season, game_type):
        """Save all games of one type for one season"""
        if not season_events:
            return
        
        season_dir = self.output_dir / f"season_{season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(season_events)
        df = self._calculate_additional_features(df)
        
        combined_path = season_dir / f"all_{game_type}_games.csv"
        df.to_csv(combined_path, index=False)
        print(f"    Saved {len(df)} events to {combined_path}")
    
    def _process_single_game(self, json_file, season, game_type):
        """Take one JSON file and get the shot/goal events out of it"""
        with open(json_file, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        game_id = json_file.stem
        events = []
        plays = game_data.get('plays', [])
        
        game_info = self._extract_game_metadata(game_data, game_id, season, game_type)
        
        for play in plays:
            event_type = play.get('typeDescKey', '')
            if event_type not in ['shot-on-goal', 'goal']:
                continue
            
            event_data = game_info.copy()
            event_data.update(self._extract_event_features(play, event_type))
            events.append(event_data)
        
        return events
    
    def _extract_game_metadata(self, game_data, game_id, season, game_type):
        """Get basic info about the game"""
        return {
            'game_id': game_id,
            'season': season,
            'game_type': game_type,
            'game_date': game_data.get('gameDate', ''),
            'away_team': game_data.get('awayTeam', {}).get('name', ''),
            'home_team': game_data.get('homeTeam', {}).get('name', '')
        }
    
    def _extract_event_features(self, play, event_type):
        """Pick out the important details from a play"""
        details = play.get('details', {})
        period_info = play.get('periodDescriptor', {})
        
        event_data = {
            'period': period_info.get('number', 1),
            'period_type': period_info.get('periodType', 'REG'),
            'period_time': play.get('timeInPeriod', ''),
            'time_remaining': play.get('timeRemaining', ''),
            'event_type': event_type.upper().replace('-', '_'),
            'team_id': details.get('eventOwnerTeamId', ''),
            'team_name': self.team_mapping.get(str(details.get('eventOwnerTeamId', '')), 'Unknown'),
            'x_coord': details.get('xCoord'),
            'y_coord': details.get('yCoord'),
            'zone_code': details.get('zoneCode', ''),
            'shot_type': details.get('shotType', ''),
            'shooter_name': self._get_player_name(details, 'shootingPlayerId' if event_type == 'shot-on-goal' else 'scoringPlayerId'),
            'goalie_name': self._get_player_name(details, 'goalieInNetId'),
        }
        
        if event_type == 'goal':
            assists = []
            for i in range(1, 3):
                assist_id = details.get(f'assist{i}PlayerId')
                if assist_id:
                    assists.append(self._get_player_name(details, f'assist{i}PlayerId'))
            
            event_data.update({
                'empty_net': details.get('emptyNet', False),
                'strength': details.get('strength', 'EVEN'),
                'game_winning_goal': details.get('gameWinningGoal', False),
                'scorer_name': event_data['shooter_name'],
                'assists': ', '.join(assists) if assists else ''
            })
        else:
            event_data.update({
                'empty_net': False,
                'strength': '',
                'game_winning_goal': False,
                'scorer_name': '',
                'assists': ''
            })
        
        return event_data
    
    def _get_player_name(self, details, player_id_key):
        """Just return Player_ID since names aren’t in JSON"""
        player_id = details.get(player_id_key)
        if player_id:
            return f"Player_{player_id}"
        return ''
    
    def _calculate_additional_features(self, df):
        """Add some extra stats like distance, angle, and seconds"""
        df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
        df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
        
        valid_coords = df['x_coord'].notna() & df['y_coord'].notna()
        df.loc[valid_coords, 'distance_from_net'] = (
            (df.loc[valid_coords, 'x_coord'] - 89) ** 2 + 
            df.loc[valid_coords, 'y_coord'] ** 2
        ) ** 0.5
        
        df['is_goal'] = (df['event_type'] == 'GOAL').astype(int)
        
        df.loc[valid_coords, 'shot_angle'] = np.degrees(
            np.arctan2(np.abs(df.loc[valid_coords, 'y_coord']), 
                      89 - df.loc[valid_coords, 'x_coord'])
        )
        
        def time_to_seconds(time_str):
            if pd.isna(time_str) or time_str == '':
                return 0
            try:
                minutes, seconds = map(int, time_str.split(':'))
                return minutes * 60 + seconds
            except:
                return 0
        
        df['period_time_seconds'] = df['period_time'].apply(time_to_seconds)
        df['game_time_seconds'] = (df['period'] - 1) * 1200 + df['period_time_seconds']
        
        return df
    
    def discuss_strength_feature(self):
        """Q4.2: Explain how to add strength info and extra features"""
        discussion = """
Right now, the JSON only gives us strength info for goals (like power plays), but not for shots.
To fix that, we would need to:

1. Keep track of penalties to know which players are missing.
2. Count how many skaters are on the ice for each team.
3. Compare the numbers (5v4, 5v3, 6v5, etc.) and assign to all events.
4. Apply to both shots and goals so everything has correct strength.

Extra features we could add:
- Rebound shots (within 3s and 20 feet of another shot by same team).
- Rush shots (within 5s of zone entry).
- High-danger shots (slot area, deflections, cross-crease passes).

This would make our data better for studying shot quality and scoring chances.
"""
        print(discussion)
        return discussion
    
    def show_dataframe_sample(self, df, n=10):
        """Show a little piece of the dataframe to check it looks good"""
        if df.empty:
            print("No data found")
            return
        
        print("Here’s a small sample of the tidy dataframe:")
        display_columns = [
            'game_id', 'season', 'event_type', 'period', 'period_time', 
            'team_name', 'shooter_name', 'shot_type', 'is_goal', 'distance_from_net'
        ]
        sample = df[[col for col in display_columns if col in df.columns]].head(n)
        print(sample.to_string(index=False))
        return sample

def main():
    """Run everything for Task 4"""
    creator = NHLTidyDataCreator("data/raw", "data/tidy")
    
    print("TASK 4: TIDY DATA CREATION")
    
    print("Step 1: Making tidy data...")
    df = creator.create_tidy_dataframe(seasons=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    
    if not df.empty:
        print("\nStep 2: Showing dataframe sample...")
        creator.show_dataframe_sample(df)
        
        print("\nStep 3: Writing discussion for Q4.2...")
        discussion = creator.discuss_strength_feature()
        
        with open(creator.output_dir / "q4_2_discussion.txt", "w") as f:
            f.write(discussion)
        
        print("\nTask 4 done!")
    else:
        print("Could not build dataframe")

if __name__ == "__main__":
    main()
