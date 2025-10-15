"""
Tidy Data Creator
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

class NHLTidyDataCreator:
    def __init__(self, processed_data_dir=None, output_dir=None, raw_data_dir=None):
        if processed_data_dir is None:
            self.processed_data_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\processed")
        else:
            self.processed_data_dir = Path(processed_data_dir)
            
        if output_dir is None:
            self.output_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\tidy")
        else:
            self.output_dir = Path(output_dir)

        if raw_data_dir is None:
            self.raw_data_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\raw")
        else:
            self.raw_data_dir = Path(raw_data_dir)
        
        print(f"Raw data directory: {self.raw_data_dir}")
        print(f"Directory exists: {self.raw_data_dir.exists()}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build player name mapping from raw files
        self.player_mapping, self.goalie_mapping = self._build_player_name_mapping()

    def _build_player_name_mapping(self):
        """Build player name mapping from rosterSpots with correct structure"""
        print("Building player name mapping from raw JSON files...")
        player_mapping = {}
        goalie_mapping = {}
        
        if not self.raw_data_dir.exists():
            print(f"Raw data directory not found: {self.raw_data_dir}")
            return player_mapping, goalie_mapping
        
        json_files = list(self.raw_data_dir.rglob("*.json"))
        print(f"Found {len(json_files)} JSON files to scan")
        
        for json_file in tqdm(json_files, desc="Scanning JSON files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                
                if 'rosterSpots' in game_data:
                    for player in game_data['rosterSpots']:
                        player_id = player.get('playerId')
                        first_name = player.get('firstName', {}).get('default', '')
                        last_name = player.get('lastName', {}).get('default', '')
                        full_name = f"{first_name} {last_name}".strip()
                        position = player.get('positionCode', '')
                        
                        if player_id and full_name and full_name != " ":
                            if position == 'G':
                                goalie_mapping[player_id] = full_name
                            else:
                                player_mapping[player_id] = full_name
                
                for event in game_data.get('plays', []):
                    for player in event.get('players', []):
                        player_id = player.get('playerId')
                        player_info = player.get('player', {})
                        first_name = player_info.get('firstName', {}).get('default', '')
                        last_name = player_info.get('lastName', {}).get('default', '')
                        full_name_alt = f"{first_name} {last_name}".strip()
                        full_name = player_info.get('fullName', {}).get('default', full_name_alt)
                        
                        if player_id and full_name and full_name != " ":
                            player_type = player.get('playerType', '').lower()
                            if 'goalie' in player_type:
                                goalie_mapping[player_id] = full_name
                            else:
                                player_mapping[player_id] = full_name
                                
            except Exception:
                continue
        
        print(f"Loaded {len(player_mapping)} player names and {len(goalie_mapping)} goalie names")
        return player_mapping, goalie_mapping

    def _replace_ids_with_real_names(self, df):
        """Replace Player_XXXXXXX and Goalie_XXXXXXX with real names"""
        print("Replacing player IDs with real names...")
        
        def extract_id(id_string):
            if pd.isna(id_string) or id_string in ['Unknown', 'No Goalie']:
                return None
            try:
                if isinstance(id_string, str):
                    if id_string.startswith('Player_'):
                        return int(id_string.replace('Player_', ''))
                    elif id_string.startswith('Goalie_'):
                        return int(id_string.replace('Goalie_', ''))
                return None
            except:
                return None
        
        player_names = []
        for player_id_str in df['player_name']:
            player_id = extract_id(player_id_str)
            if player_id and player_id in self.player_mapping:
                player_names.append(self.player_mapping[player_id])
            else:
                player_names.append(player_id_str)
        
        goalie_names = []
        for goalie_id_str in df['goalie_name']:
            goalie_id = extract_id(goalie_id_str)
            if goalie_id and goalie_id in self.goalie_mapping:
                goalie_names.append(self.goalie_mapping[goalie_id])
            else:
                goalie_names.append(goalie_id_str)
        
        df['player_name'] = player_names
        df['goalie_name'] = goalie_names
        
        print("Player and goalie name replacement completed.")
        return df

    def _determine_home_away_teams(self, df):
        """Determine home and away teams for each game"""
        print("Determining home and away teams...")
        game_teams = {}
        
        for game_id in df['game_id'].unique():
            game_data = df[df['game_id'] == game_id]
            teams = game_data['team_id'].unique()
            
            if len(teams) != 2:
                continue
            
            period1_data = game_data[game_data['period'] == 1]
            if len(period1_data) < 10:
                period1_data = game_data
            
            team_stats = {}
            for team in teams:
                team_shots = period1_data[period1_data['team_id'] == team]
                if len(team_shots) == 0:
                    team_stats[team] = {'left_ratio': 0.5}
                    continue
                
                right_shots = len(team_shots[team_shots['x_coord'] > 0])
                left_shots = len(team_shots[team_shots['x_coord'] < 0])
                total_shots = right_shots + left_shots
                left_ratio = left_shots / total_shots if total_shots > 0 else 0.5
                team_stats[team] = {'left_ratio': left_ratio}
            
            teams_sorted = sorted(teams, key=lambda x: team_stats[x]['left_ratio'], reverse=True)
            game_teams[game_id] = {
                'home_team': teams_sorted[0],
                'away_team': teams_sorted[1]
            }
        
        print(f"Processed {len(game_teams)} games")
        return game_teams

    def _get_attacking_net(self, team_id, period, game_teams, game_id):
        if game_id not in game_teams:
            return 'left' if period % 2 == 1 else 'right'
        
        home_team = game_teams[game_id]['home_team']
        is_home = team_id == home_team
        
        if is_home:
            return 'left' if period % 2 == 1 else 'right'
        else:
            return 'right' if period % 2 == 1 else 'left'

    def _calculate_distance_and_angle(self, x, y, net_side):
        if pd.isna(x) or pd.isna(y):
            return None, None
        
        net_x, net_y = (89, 0) if net_side == 'right' else (-89, 0)
        dx = x - net_x
        dy = y - net_y
        distance = (dx**2 + dy**2) ** 0.5
        
        adjacent = abs(x - net_x)
        opposite = abs(y - net_y)
        
        if adjacent == 0:
            angle = 90.0
        else:
            angle_rad = np.arctan(opposite / adjacent)
            angle = np.degrees(angle_rad)
        
        return distance, angle

    def _transform_coordinates(self, df):
        df_transformed = df.copy()
        for idx, row in df_transformed.iterrows():
            if row['attacking_net'] == 'left':
                df_transformed.at[idx, 'x_coord'] = -row['x_coord']
                df_transformed.at[idx, 'y_coord'] = -row['y_coord']
        return df_transformed

    def _calculate_shot_features(self, df):
        print("Calculating shot features...")
        
        df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
        df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
        
        game_teams = self._determine_home_away_teams(df)
        attacking_sides, team_types = [], []
        
        for _, row in df.iterrows():
            attacking_side = self._get_attacking_net(
                row['team_id'], row['period'], game_teams, row['game_id']
            )
            attacking_sides.append(attacking_side)
            
            if row['game_id'] in game_teams:
                is_home = row['team_id'] == game_teams[row['game_id']]['home_team']
                team_types.append('home' if is_home else 'away')
            else:
                team_types.append('unknown')
        
        df['attacking_net'] = attacking_sides
        df['team_type'] = team_types
        
        distances, angles = [], []
        for _, row in df.iterrows():
            distance, angle = self._calculate_distance_and_angle(
                row['x_coord'], row['y_coord'], row['attacking_net']
            )
            distances.append(distance)
            angles.append(angle)
        
        df['distance_from_net'] = distances
        df['shot_angle'] = angles
        
        df = self._transform_coordinates(df)
        return df, game_teams

    def _add_time_features(self, df):
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return 0
            try:
                minutes, seconds = time_str.split(':')
                return int(minutes) * 60 + int(seconds)
            except:
                return 0
        
        df['period_time_seconds'] = df['period_time'].apply(time_to_seconds)
        return df

    def create_tidy_dataframe(self, seasons=None):
        if seasons is None:
            seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

        all_seasons_data = []

        for season in seasons:
            print(f"Processing season {season}")
            for game_type in ['general', 'playoff']:
                season_dir = self.processed_data_dir / f"season_{season}" / game_type
                if not season_dir.exists():
                    print(f"Skipping {game_type} (no data)")
                    continue

                combined_file = season_dir / f"all_{game_type}_games.csv"
                if combined_file.exists():
                    try:
                        season_df = pd.read_csv(combined_file)
                        print(f"Loaded {len(season_df)} events from {game_type}")
                        
                        season_df = self._replace_ids_with_real_names(season_df)
                        season_df, game_teams = self._calculate_shot_features(season_df)
                        season_df = self._add_time_features(season_df)
                        
                        season_output_dir = self.output_dir / f"season_{season}"
                        season_output_dir.mkdir(parents=True, exist_ok=True)
                        season_df.to_csv(season_output_dir / f"all_{game_type}_games_tidy.csv", index=False)
                        
                        all_seasons_data.append(season_df)
                        print(f"Processed {len(season_df)} events for {season} {game_type}")
                        
                    except Exception as e:
                        print(f"Error processing {combined_file}: {e}")
                else:
                    print(f"No combined file found for {season} {game_type}")

        if not all_seasons_data:
            print("No data processed.")
            return pd.DataFrame()

        print("Creating final dataset...")
        df = pd.concat(all_seasons_data, ignore_index=True)
        combined_path = self.output_dir / "all_seasons_combined.csv"
        df.to_csv(combined_path, index=False)
        print(f"Saved combined data to {combined_path}")

        sample_path = self.output_dir / "tidy_data_sample.csv"
        df.head(100).to_csv(sample_path, index=False)
        print(f"Saved sample to {sample_path}")

        self._validate_data(df)
        return df

    def _validate_data(self, df):
        print("Data Validation Summary")
        print(f"Total events: {len(df)}")
        
        real_players = ~df['player_name'].str.startswith('Player_', na=False)
        real_goalies = ~df['goalie_name'].str.startswith('Goalie_', na=False)
        
        print(f"Real player names: {real_players.sum()}/{len(df)}")
        print(f"Real goalie names: {real_goalies.sum()}/{len(df)}")
        
        valid_coords = df['x_coord'].notna() & df['y_coord'].notna()
        print(f"Valid coordinates: {valid_coords.sum()}")
        
        goals = df['is_goal'].sum()
        shots = (df['event_type'] == 'SHOT_ON_GOAL').sum()
        print(f"Goals: {goals}")
        print(f"Shots: {shots}")
        print(f"Shooting %: {(goals / shots * 100 if shots > 0 else 0):.1f}%")

        return df

def main():
    print("Creating tidy data with real player names...")
    creator = NHLTidyDataCreator()
    df = creator.create_tidy_dataframe(seasons=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    
    if not df.empty:
        print("Processing completed successfully.")
        print(f"Final shape: {df.shape}")
        print(f"Output directory: {creator.output_dir}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
