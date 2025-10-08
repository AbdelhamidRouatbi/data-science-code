"""
Task 4: Create tidy data with coordinate transformations and distance calculations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

class NHLTidyDataCreator:
    """Makes clean NHL data with distance and angle values"""

    def __init__(self, processed_data_dir=None, output_dir=None):
        # Set input and output folders
        if processed_data_dir is None:
            self.processed_data_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\processed")
        else:
            self.processed_data_dir = Path(processed_data_dir)
            
        if output_dir is None:
            self.output_dir = Path(r"C:\Users\AgeTeQ\Desktop\data\classes\DS\tp1\project-template\data\tidy")
        else:
            self.output_dir = Path(output_dir)
        
        print(f"Processed data directory: {self.processed_data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _assign_net_by_majority(self, df, team_id, period):
        """Find which net a team is shooting at in each period"""
        team_period_shots = df[(df['team_id'] == team_id) & (df['period'] == period)]
        
        if len(team_period_shots) == 0:
            return 'right'
        
        right_side_shots = len(team_period_shots[team_period_shots['x_coord'] > 0])
        left_side_shots = len(team_period_shots[team_period_shots['x_coord'] < 0])
        
        print(f"Team {team_id}, Period {period}: {right_side_shots} right shots, {left_side_shots} left shots")
        
        return 'right' if right_side_shots > left_side_shots else 'left'

    def _calculate_distance_and_angle(self, x, y, net_side):
        """Calculate distance and angle for a shot"""
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

    def _transform_coordinates_for_visualization(self, df, team_net_assignments):
        """Flip coordinates so all shots face the same net"""
        df_transformed = df.copy()
        left_net_shots = 0
        
        for idx, row in df_transformed.iterrows():
            key = (row['team_id'], row['period'])
            net_side = team_net_assignments.get(key, 'right')
            
            if net_side == 'left':
                df_transformed.at[idx, 'x_coord'] = abs(row['x_coord'])
                df_transformed.at[idx, 'y_coord'] = -row['y_coord']
                left_net_shots += 1
        
        print(f"Transformed {left_net_shots} shots to right side")
        return df_transformed

    def _calculate_shot_features(self, df):
        """Add distance and angle columns"""
        print("Calculating shot features...")
        
        df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
        df['y_coord'] = pd.to_numeric(df['y_coord'], errors='coerce')
        
        team_net_assignments = {}
        team_period_combinations = df[['team_id', 'period']].drop_duplicates()
        
        print("Finding net sides for all teams...")
        for _, combo in team_period_combinations.iterrows():
            team_id = combo['team_id']
            period = combo['period']
            net_side = self._assign_net_by_majority(df, team_id, period)
            team_net_assignments[(team_id, period)] = net_side
            print(f"  Team {team_id}, Period {period}: {net_side} net")
        
        distances, angles = [], []
        for _, row in df.iterrows():
            key = (row['team_id'], row['period'])
            net_side = team_net_assignments.get(key, 'right')
            distance, angle = self._calculate_distance_and_angle(row['x_coord'], row['y_coord'], net_side)
            distances.append(distance)
            angles.append(angle)
        
        df['distance_from_net'] = distances
        df['shot_angle'] = angles
        df = self._transform_coordinates_for_visualization(df, team_net_assignments)
        
        return df

    def _add_time_features(self, df):
        """Add time in seconds for each event"""
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return 0
            try:
                minutes, seconds = time_str.split(':')
                return int(minutes) * 60 + int(seconds)
            except:
                return 0
        
        df['period_time_seconds'] = df['period_time'].apply(time_to_seconds)
        df['game_time_seconds'] = (df['period'] - 1) * 1200 + df['period_time_seconds']
        return df

    def create_tidy_dataframe(self, seasons=None):
        """Create the full tidy dataset"""
        if seasons is None:
            seasons = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

        all_seasons_data = []

        for season in seasons:
            print(f"\n=== Processing season {season} ===")
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
                        
                        season_df = self._calculate_shot_features(season_df)
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

        print("\nCreating final dataset...")
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
        """Print summary of the dataset"""
        print("\n=== Data Validation ===")
        print(f"Total events: {len(df)}")
        
        valid_coords = df['x_coord'].notna() & df['y_coord'].notna()
        print(f"Valid coordinates: {valid_coords.sum()}")
        
        right_side = (df['x_coord'] > 0).sum()
        left_side = (df['x_coord'] < 0).sum()
        print(f"Right side shots: {right_side}")
        print(f"Left side shots: {left_side}")
        
        goals = df['is_goal'].sum()
        shots = (df['event_type'] == 'SHOT_ON_GOAL').sum()
        print(f"Goals: {goals}")
        print(f"Shots: {shots}")
        print(f"Shooting %: {(goals / shots * 100 if shots > 0 else 0):.1f}%")

        if 'distance_from_net' in df.columns:
            valid_distances = df['distance_from_net'].notna()
            if valid_distances.any():
                print(f"Distance range: {df[valid_distances]['distance_from_net'].min():.1f}–{df[valid_distances]['distance_from_net'].max():.1f}")
            
            valid_angles = df['shot_angle'].notna()
            if valid_angles.any():
                print(f"Angle range: {df[valid_angles]['shot_angle'].min():.1f}–{df[valid_angles]['shot_angle'].max():.1f}")

        return df

def main():
    """Run the data creation"""
    print("Task 4: Creating tidy data")
    creator = NHLTidyDataCreator()
    df = creator.create_tidy_dataframe(seasons=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    
    if not df.empty:
        print("\n=== Done ===")
        print(f"Final shape: {df.shape}")
        print(f"Output: {creator.output_dir}")
        print("\nSample of tidy data:")
        print(df[['game_id', 'team_name', 'x_coord', 'y_coord', 'distance_from_net', 'shot_angle', 'is_goal']].head(10))
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
