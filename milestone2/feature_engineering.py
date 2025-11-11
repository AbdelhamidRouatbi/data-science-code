import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import wandb

class FeatureEngineering:

    def __init__(self, tidy_data_path_csv = "data/tidy/all_seasons_combined.csv",
                 raw_data_path = "data/raw/",
                 save_data_path = "data/milestone2/"
                ):
        self._tidy_data_path_csv = tidy_data_path_csv
        self._raw_data_path = raw_data_path
        self._save_data_path = save_data_path

        self._cached_game = None

    def _split_data(self):
        try:
            df = pd.read_csv(self._tidy_data_path_csv)
        except FileNotFoundError:
            print(f"Error: tidy data csv not found in {self._tidy_data_path_csv}")
        df_train = df[df["season"].isin([2016, 2017, 2018, 2019])].copy()
        df_test = df[df["season"] == 2020].copy()
        del df
        return df_train, df_test

    def feature_engineering_1(self, df):
        features = ["distance_from_net", "shot_angle", "empty_net", "is_goal"]
        df["empty_net"] = df["goalie_name"].isna().astype(int)
        df = df[features].copy()
        return df

    def _load_raw_game(self, game_id):
        # load raw game json in cache
        season = str(game_id)[:4]
        is_regular = str(game_id)[4:6] == "02"
        if (is_regular):
            game_type = "general"
        else:
            game_type = "playoff"
        json_path = self._raw_data_path + "season_" + season + "/" + game_type + "/" + str(game_id) + ".json"
        with open(json_path, "r") as f:
            self._cached_game = json.load(f)

    def _to_seconds(self, t):
        if t is None:
            return None
        m, s = map(int, t.split(":"))
        return m * 60 + s

    def _get_last_event(self, row):
        previous_events = [
            play for play in self._cached_game["plays"] if (
                (self._to_seconds(play.get("timeInPeriod")) <= row["period_time_seconds"]) and
                (play.get("periodDescriptor").get("number") == 1)
            )
        ]
        try:
            previous_event = previous_events[-2]
            current_event = previous_events[-1]
        except IndexError:
            previous_event = None
            current_event = None
        return previous_events

    def _format_event(self, event):
        if not isinstance(event, dict):
            return {
                "typeDescKey": None,
                "details": {"xCoord": None, "yCoord": None},
                "timeInPeriod": None,
            }
        details = event.get("details", {}) or {}
        return {
            "typeDescKey": event.get("typeDescKey"),
            "situationCode": event.get("situationCode"),
            "details": {
                "xCoord": details.get("xCoord"),
                "yCoord": details.get("yCoord"),
                "teamId": details.get("eventOwnerTeamId")
            },
            "timeInPeriod": event.get("timeInPeriod"),
        }
    
    def feature_engineering_2(self, df):
        def compute_angle(x,y):
            if x is None or y is None:
                return None
            y = abs(y)
            net_right = (89, 0)
            net_left = (-89, 0)
            dist_right = np.hypot(x - net_right[0], y - net_right[1])
            dist_left = np.hypot(x - net_left[0], y - net_left[1])
            net_x, net_y = (net_right if dist_right < dist_left else net_left)
            angle = np.degrees(np.arctan2(net_y-y, net_x-x))
            return angle

        def is_powerplay(event):
            try:
                situationCode = event["situationCode"]
                return situationCode[1] != situationCode[2]
            except:
                return False

        def compute_time_since_powerplay(previous_events):
            try:
                current_event = previous_events[-1]
                current_time = self._to_seconds(current_event["timeInPeriod"])
                powerplay_start_event = current_event
                if not is_powerplay(current_event):
                    return 0
                for event in reversed(previous_events):
                    if not is_powerplay(event):
                        powerplay_start_event = event
                        break
                powerplay_start_time = self._to_seconds(powerplay_start_event["timeInPeriod"])
                return current_time - powerplay_start_time
            except:
                return 0
            
        df["empty_net"] = df["goalie_name"].isna().astype(int)
        df["last_event_type"] = None
        df["last_event_x"] = None
        df["last_event_y"] = None
        df["time_since_last_event"] = None
        df["distance_from_last_event"] = None
        df["rebound"] = False
        df["angle_change"] = 0.0
        df["event_speed"] = 0.0
        df["friendly_player_count"] = 5
        df["opponent_player_count"] = 5
        df["player_count_diff"] = 0
        df["time_since_powerplay"] = 0
        grouped = df.groupby("game_id")
        for game_id, group in tqdm(grouped, desc="Applying feature engineering 2"):
            self._load_raw_game(game_id)
            home_team = self._cached_game["homeTeam"]["id"]
            away_team = self._cached_game["awayTeam"]["id"]
            for _, row in group.iterrows():
                previous_events= self._get_last_event(row)
                try:
                    current_event = previous_events[-1]
                except:
                    current_event = None
                try:
                    last_event = previous_events[-2]
                except:
                    last_event = None
                    
                current_event = self._format_event(current_event)
                last_event = self._format_event(last_event)
                
                last_event_type = last_event["typeDescKey"]
                last_event_xcoord = last_event["details"]["xCoord"]
                last_event_ycoord = last_event["details"]["yCoord"]
                try:
                    time_since_last_event = row["period_time_seconds"] - self._to_seconds(last_event["timeInPeriod"])
                except:
                    time_since_last_event = None 
                distance_from_last_event = None
                if (last_event_xcoord is not None and last_event_ycoord is not None):
                    distance_from_last_event = math.hypot(
                        row["x_coord"] - last_event_xcoord,
                        row["y_coord"] - last_event_ycoord
                    )

                try: 
                    rebound = (
                        "shot" in last_event_type.lower() and
                        last_event_type.lower() != "blocked-shot" and 
                        current_event["details"]["teamId"] == last_event["details"]["teamId"]
                    )
                except:
                    rebound = False

                angle_change = 0
                if (rebound):
                    last_event_angle = compute_angle(last_event_xcoord, last_event_ycoord)
                    diff = abs(row["shot_angle"] - last_event_angle)
                    if diff > 180:
                        diff = 360 - diff
                    angle_change = diff
                    
                try:
                    event_speed = distance_from_last_event/time_since_last_event
                except:
                    event_speed = 0
                    
                try:
                    situation_code = current_event["situationCode"]
                    if current_event["details"]["teamId"] == home_team:
                        friendly_player_count = int(situation_code[2])
                        opponent_player_count = int(situation_code[1])
                    elif current_event["details"]["teamId"] == away_team:
                        friendly_player_count = int(situation_code[1])
                        opponent_player_count = int(situation_code[2])
                except:
                    friendly_player_count = 5
                    opponent_player_count = 5

                time_since_powerplay = compute_time_since_powerplay(previous_events)

                values = {
                    "last_event_type": last_event_type,
                    "last_event_x": last_event_xcoord,
                    "last_event_y": last_event_ycoord,
                    "time_since_last_event": time_since_last_event,
                    "last_event_distance": distance_from_last_event,
                    "rebound": rebound,
                    "angle_change": angle_change,
                    "event_speed": event_speed,
                    "friendly_player_count": friendly_player_count,
                    "opponent_player_count": opponent_player_count,
                    "player_count_diff": friendly_player_count - opponent_player_count,
                    "time_since_powerplay": time_since_powerplay,
                }

                for col, val in values.items():
                    df.at[row.name, col] = val
        features = [
            "period_time_seconds", "period", "x_coord", "y_coord",
            "distance_from_net", "shot_angle", "shot_type", "empty_net", "last_event_type",
            "last_event_x", "last_event_y", "time_since_last_event", "last_event_distance",
            "rebound", "angle_change", "event_speed", "friendly_player_count", "opponent_player_count", "player_count_diff", "time_since_powerplay", "is_goal"
        ]
        return df[features].copy()

    def create_data(self):
        # --- split data --- #
        print("splitting data ...")
        df_train, df_test = self._split_data()
        print("splitting data ✅")

        # Define paths
        baseline_train_path = os.path.join(self._save_data_path, "baseline_train.csv")
        advanced_train_path = os.path.join(self._save_data_path, "advanced_train.csv")
        test_path = os.path.join(self._save_data_path, "test.csv")
        test_baseline_general_path = os.path.join(self._save_data_path, "baseline_general_test.csv")
        test_baseline_playoff_path = os.path.join(self._save_data_path, "baseline_playoff_test.csv")
        test_advanced_general_path = os.path.join(self._save_data_path, "advanced_general_test.csv")
        test_advanced_playoff_path = os.path.join(self._save_data_path, "advanced_playoff_test.csv")

        os.makedirs(os.path.dirname(self._save_data_path), exist_ok=True)

        # --- feature engineering 1 --- #
        if not os.path.exists(baseline_train_path):
            print("feature engineering 1 ...")
            df_train_1 = self.feature_engineering_1(df_train)
            print("feature engineering 1 ✅")

            print("saving test data and baseline training data ...")
            df_train_1.to_csv(baseline_train_path, index=False)
            df_test.to_csv(test_path, index=False)
            print("saving test data and baseline training data ✅")
        else:
            print("baseline_train.csv already exists, skipping feature engineering 1")
            df_train_1 = pd.read_csv(baseline_train_path)
            df_test = pd.read_csv(test_path)

        # --- feature engineering 2 --- #
        if not os.path.exists(advanced_train_path):
            print("feature engineering 2 ...")
            df_train_2 = self.feature_engineering_2(df_train)
            print("feature engineering 2 ✅")
            print("saving advanced_train.csv ...")
            df_train_2.to_csv(advanced_train_path, index=False)
            print("saving advanced_train.csv ✅")
        else:
            print("advanced_train.csv already exists, skipping feature engineering 2")
            df_train_2 = pd.read_csv(advanced_train_path)

        # --- feature engineering for test data --- #
        if not all(os.path.exists(p) for p in [
            test_baseline_general_path,
            test_baseline_playoff_path,
            test_advanced_general_path,
            test_advanced_playoff_path
        ]):
            print("feature engineering test data...")

            # Split by game type
            df_test_general = df_test[df_test["game_type"] == "general"].copy()
            df_test_playoff = df_test[df_test["game_type"] == "playoff"].copy()

            # Apply feature engineering
            df_test_general_baseline = self.feature_engineering_1(df_test_general)
            df_test_playoff_baseline = self.feature_engineering_1(df_test_playoff)
            df_test_general_advanced = self.feature_engineering_2(df_test_general)
            df_test_playoff_advanced = self.feature_engineering_2(df_test_playoff)

            print("feature engineering test data ✅")

            # Save results
            print("saving feature engineered test data...")
            df_test_general_baseline.to_csv(test_baseline_general_path, index=False)
            df_test_playoff_baseline.to_csv(test_baseline_playoff_path, index=False)
            df_test_general_advanced.to_csv(test_advanced_general_path, index=False)
            df_test_playoff_advanced.to_csv(test_advanced_playoff_path, index=False)
            print("saving feature engineered test data ✅")
        else:
            print("Feature-engineered test CSVs already exist, skipping test feature engineering.")                


    def generate_winnipeg_washington_df(self):
        df = pd.read_csv(self._tidy_data_path_csv)
        df = df[df["game_id"] == 2017021065].copy()
        df = self.feature_engineering_2(df)
        print("")
        run = wandb.init(project="milestone_2")
        # create a wandb Artifact for each meaningful step
        artifact = wandb.Artifact(
            "wpg_v_wsh_2017021065",
            type="dataset"
            )
        # add data
        my_table = wandb.Table(dataframe=df)
        artifact.add(my_table, "wpg_v_wsh_2017021065")
        run.log_artifact(artifact)        



def main():
    fe = FeatureEngineering()
    fe.create_data()
    #fe.generate_winnipeg_washington_df()

if __name__ == "__main__":
    main()
