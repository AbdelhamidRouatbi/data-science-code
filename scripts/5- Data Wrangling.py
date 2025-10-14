import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class NHLDataWrangler:
    def __init__(self, raw_data_dir="data/raw", clean_data_dir="data/clean"):
        self.raw_data_dir = Path(raw_data_dir)
        self.clean_data_dir = Path(clean_data_dir)
        self.clean_data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_json_with_fallback(self, json_file):
        """
        Load a JSON file with several possible encodings.
        If none of the encodings work, it tries a binary read.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(json_file, 'r', encoding=encoding, errors='replace') as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except Exception:
                continue
        
        try:
            with open(json_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
                return json.loads(content)
        except:
            print(f"Could not read {json_file.name}")
            return None

    def create_tidy_datasets(self):
        """
        Main method: goes through all season folders and creates plays.csv and players.csv
        """
        print("Creating tidy datasets...")
        
        all_plays = []
        all_players = {}
        failed_files = []
        
        season_folders = [f for f in self.raw_data_dir.iterdir() 
                         if f.is_dir() and f.name.startswith("season_")]
        
        for season_folder in season_folders:
            season = season_folder.name.replace("season_", "")
            print(f"\nProcessing season {season}...")
            
            for game_type in ["general", "playoff"]:
                game_folder = season_folder / game_type
                if game_folder.exists():
                    json_files = list(game_folder.glob("*.json"))
                    print(f"  {game_type}: {len(json_files)} games")
                    
                    plays, players, failed = self._process_season_games(game_folder, season, game_type)
                    all_plays.extend(plays)
                    all_players.update(players)
                    failed_files.extend(failed)
        
        print(f"Processed {len(all_plays)} plays in total")
        print(f"Collected {len(all_players)} players")
        print(f"Failed to process {len(failed_files)} files")
        
        plays_df = self._create_plays_dataframe(all_plays)
        players_df = self._create_players_dataframe(all_players)
        
        plays_df = self._handle_missing_values(plays_df)
        players_df = self._handle_missing_values(players_df)
        
        plays_df.to_csv(self.clean_data_dir / "plays.csv", index=False)
        players_df.to_csv(self.clean_data_dir / "players.csv", index=False)
        
        if failed_files:
            with open(self.clean_data_dir / "failed_files.txt", 'w') as f:
                for file_path in failed_files:
                    f.write(f"{file_path}\n")
        
        print(f"Saved {len(plays_df)} plays to plays.csv")
        print(f"Saved {len(players_df)} players to players.csv")
        
        return plays_df, players_df
    
    def _process_season_games(self, game_folder, season, game_type):
        plays = []
        players = {}
        failed_files = []
        
        json_files = list(game_folder.glob("*.json"))
        
        for json_file in tqdm(json_files, desc=f"    {game_type}"):
            game_data = self._load_json_with_fallback(json_file)
            
            if game_data is None:
                failed_files.append(json_file)
                continue
            
            try:
                game_id = json_file.stem
                game_plays, game_players = self._extract_game_data(game_data, game_id, season, game_type)
                plays.extend(game_plays)
                players.update(game_players)
                
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                failed_files.append(json_file)
                continue
        
        return plays, players, failed_files
    
    def _extract_game_data(self, game_data, game_id, season, game_type):
        plays = []
        players = {}
        
        for player in game_data.get("rosterSpots", []):
            try:
                player_id = str(player.get("playerId", ""))
                if player_id and player_id not in players:
                    players[player_id] = {
                        'player_id': player_id,
                        'first_name': player.get("firstName", {}).get("default", ""),
                        'last_name': player.get("lastName", {}).get("default", ""),
                        'position': player.get("positionCode", ""),
                        'team_id': player.get("teamId")
                    }
            except:
                continue
        
        for play in game_data.get("plays", []):
            try:
                play_data = self._extract_play_data(play, game_id, season, game_type)
                if play_data:
                    plays.append(play_data)
            except:
                continue
        
        return plays, players
    
    def _extract_play_data(self, play, game_id, season, game_type):
        try:
            details = play.get("details", {})
            event_type = play.get("typeDescKey", "")
            
            if event_type not in ["shot-on-goal", "goal"]:
                return None
            
            play_data = {
                'game_id': game_id,
                'season': season,
                'game_type': game_type,
                'event_type': event_type.upper().replace("-", "_"),
                'period': play.get("periodDescriptor", {}).get("number", 1),
                'period_time': play.get("timeInPeriod", ""),
                'x_coord': details.get("xCoord"),
                'y_coord': details.get("yCoord"),
                'shot_type': details.get("shotType", ""),
                'team_id': details.get("eventOwnerTeamId"),
                'player_id': str(details.get("shootingPlayerId") or details.get("scoringPlayerId", "")),
                'goalie_id': str(details.get("goalieInNetId", "")),
                'is_goal': 1 if event_type == "goal" else 0
            }
            
            return play_data
            
        except:
            return None
    
    def _create_plays_dataframe(self, plays):
        if not plays:
            return pd.DataFrame()
            
        df = pd.DataFrame(plays)
        
        numeric_cols = ['x_coord', 'y_coord', 'period', 'is_goal']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = self._calculate_play_features(df)
        
        return df
    
    def _create_players_dataframe(self, players):
        if not players:
            return pd.DataFrame()
        return pd.DataFrame(players.values())
    
    def _calculate_play_features(self, df):
        if df.empty:
            return df
            
        valid_coords = df['x_coord'].notna() & df['y_coord'].notna()
        if valid_coords.any():
            df.loc[valid_coords, 'distance_from_net'] = (
                (df.loc[valid_coords, 'x_coord'] - 89) ** 2 + 
                df.loc[valid_coords, 'y_coord'] ** 2
            ) ** 0.5
            
            df.loc[valid_coords, 'shot_angle'] = np.degrees(
                np.arctan2(np.abs(df.loc[valid_coords, 'y_coord']), 
                          89 - df.loc[valid_coords, 'x_coord'])
            )
        
        return df
    
    def _handle_missing_values(self, df):
        if df.empty:
            return df
            
        if 'game_id' in df.columns:  
            if 'x_coord' in df.columns:
                median_x = df['x_coord'].median()
                df['x_coord'] = df['x_coord'].fillna(median_x)
            if 'y_coord' in df.columns:
                median_y = df['y_coord'].median()
                df['y_coord'] = df['y_coord'].fillna(median_y)
            
            categorical_cols = ['shot_type', 'period_time']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown')
        else:
            df = df.fillna('Unknown')
        
        return df

    def create_simple_plots(self, plays_df, players_df):
        if plays_df.empty or players_df.empty:
            print("No data available for plotting")
            return None, None
        
        player_name_map = {}
        for _, row in players_df.iterrows():
            player_name = f"{row['first_name']} {row['last_name']}".strip()
            player_name_map[str(row['player_id'])] = player_name
        
        team_name_map = {
            '1': 'NJD', '2': 'NYI', '3': 'NYR', '4': 'PHI', '5': 'PIT', '6': 'BOS',
            '7': 'BUF', '8': 'MTL', '9': 'OTT', '10': 'TOR', '12': 'CAR', '13': 'FLA',
            '14': 'TBL', '15': 'WSH', '16': 'CHI', '17': 'DET', '18': 'NSH', '19': 'STL',
            '20': 'CGY', '21': 'COL', '22': 'EDM', '23': 'VAN', '24': 'ANA', '25': 'DAL',
            '26': 'LAK', '28': 'SJS', '29': 'CBJ', '30': 'MIN', '52': 'WPG', '53': 'ARI',
            '54': 'VGK', '55': 'SEA'
        }
        
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        try:
            goal_scorers = plays_df[plays_df['is_goal'] == 1].groupby('player_id').size()
            goal_scorers = goal_scorers.sort_values(ascending=False).head(10)
            
            scorer_names = []
            for player_id in goal_scorers.index:
                name = player_name_map.get(str(player_id), f"Player {player_id}")
                scorer_names.append(name)
            
            bars1 = ax1.bar(range(len(goal_scorers)), goal_scorers.values, color='skyblue', edgecolor='navy')
            ax1.set_title('Top 10 Goal Scorers (2016-2023)')
            ax1.set_xlabel('Player')
            ax1.set_ylabel('Number of Goals')
            ax1.set_xticks(range(len(goal_scorers)))
            ax1.set_xticklabels(scorer_names, rotation=45, ha='right')
            
            for bar, v in zip(bars1, goal_scorers.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(v),
                         ha='center', va='bottom')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error creating goal scorers plot:\n{e}", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Goal Scorers (Error)')
        
        try:
            team_goals = plays_df[plays_df['is_goal'] == 1].groupby('team_id').size()
            team_goals = team_goals.sort_values(ascending=False).head(10)
            
            team_names = []
            for team_id in team_goals.index:
                name = team_name_map.get(str(team_id), f"Team {team_id}")
                team_names.append(name)
            
            bars2 = ax2.bar(range(len(team_goals)), team_goals.values, color='lightcoral', edgecolor='darkred')
            ax2.set_title('Top 10 Teams by Goals (2016-2023)')
            ax2.set_xlabel('Team')
            ax2.set_ylabel('Total Goals')
            ax2.set_xticks(range(len(team_goals)))
            ax2.set_xticklabels(team_names, rotation=45, ha='right')
            
            for bar, v in zip(bars2, team_goals.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(v),
                         ha='center', va='bottom')
            
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error creating team goals plot:\n{e}", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Team Goals (Error)')
        
        plt.tight_layout()
        plt.savefig(self.clean_data_dir / 'simple_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSome quick insights:")
        print(f"Total goals: {len(plays_df[plays_df['is_goal'] == 1])}")
        print(f"Total shots: {len(plays_df[plays_df['event_type'] == 'SHOT_ON_GOAL'])}")
        print(f"Shooting percentage: {(len(plays_df[plays_df['is_goal'] == 1]) / len(plays_df) * 100):.1f}%")
        
        if 'goal_scorers' in locals() and not goal_scorers.empty:
            print(f"Top scorer: {scorer_names[0]} with {goal_scorers.values[0]} goals")
        
        if 'team_goals' in locals() and not team_goals.empty:
            print(f"Highest scoring team: {team_names[0]} with {team_goals.values[0]} goals")
        
        return goal_scorers, team_goals

def main():
    wrangler = NHLDataWrangler()
    
    plays_df, players_df = wrangler.create_tidy_datasets()
    
    if not plays_df.empty:
        wrangler.create_simple_plots(plays_df, players_df)
    
    print("Data wrangling complete.")
    return plays_df, players_df

if __name__ == "__main__":
    plays_df, players_df = main()
