import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

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
        df = df[features].copy()
        df["empty_net"] = df["empty_net"].astype(int)
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
                (self._to_seconds(play.get("timeInPeriod")) < row["period_time_seconds"]) and
                (play.get("periodDescriptor").get("number") == 1)
            )
        ]
        try:
            previous_event = previous_events[-1]
        except IndexError:
            previous_event = None
        return previous_event

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
            "details": {
                "xCoord": details.get("xCoord"),
                "yCoord": details.get("yCoord"),
            },
            "timeInPeriod": event.get("timeInPeriod"),
        }
    
    def feature_engineering_2(self, df):
        df["last_event_type"] = None
        df["last_event_x"] = None
        df["last_event_y"] = None
        df["last_event_time_seconds"] = None
        df["last_event_distance"] = None
        grouped = df.groupby("game_id")
        for game_id, group in tqdm(grouped, desc="Applying feature engineering 2"):
            self._load_raw_game(game_id)
            for _, row in group.iterrows():
                last_event = self._format_event(self._get_last_event(row))
                last_event_type = last_event["typeDescKey"]
                last_event_xcoord = last_event["details"]["xCoord"]
                last_event_ycoord = last_event["details"]["yCoord"]
                last_event_time = self._to_seconds(last_event["timeInPeriod"])
                if (last_event_xcoord is not None and last_event_ycoord is not None):
                    last_event_distance = math.hypot(
                        row["x_coord"] - last_event_xcoord,
                        row["y_coord"] - last_event_ycoord
                    )

                df.loc[row.name, "last_event_type"] = last_event_type
                df.loc[row.name, "last_event_x"] = last_event_xcoord 
                df.loc[row.name, "last_event_y"] = last_event_ycoord
                if last_event_time is not None:
                    time_diff = row["period_time_seconds"] - last_event_time
                else:
                    time_diff = None
                df.loc[row.name, "time_since_last_event"] = time_diff
                df.loc[row.name, "last_event_distance"] = last_event_distance
        return df                

    def _last_event_features(self, df):
        def compute_angle(x,y):
            if x is None or y is None:
                return None
            net_right = (89, 0)
            net_left = (-89, 0)
            dist_right = np.hypot(x - net_right[0], y - net_right[1])
            dist_left = np.hypot(x - net_left[0], y - net_left[1])
            net_x, net_y = (net_right if dist_right < dist_left else net_left)
            angle = np.degrees(np.arctan2(net_y-y, net_x-x))
            return angle
        
        def angle_change(row):
            if not row["rebound"]:
                return 0
            angle_now = row["shot_angle"]
            angle_last = compute_angle(row["last_event_x"], row["last_event_y"])
            if angle_now is None or angle_last is None:
                return None
            diff = abs(angle_now - angle_last)
            if diff > 180:
                diff = 360 - diff
            return diff

        def compute_speed(row):
            dist = row.get("last_event_distance")
            time = row.get("time_since_last_event")
            if dist is None or time is None or time == 0 or pd.isna(dist) or pd.isna(time):
                return None
            return dist / time
        df["rebound"] = df["last_event_type"].str.contains("shot", case=False, na=False)
        df["angle_change"] = df.apply(angle_change, axis=1)
        df["event_speed"] = df.apply(compute_speed, axis=1)
        return df




    def create_data(self):
        # --- split data --- #
        print("splitting data ...")
        df_train, df_test = self._split_data()
        print("splitting data ✅")

        # Define paths
        baseline_train_path = os.path.join(self._save_data_path, "baseline_train.csv")
        advanced_train_path = os.path.join(self._save_data_path, "advanced_train.csv")
        test_path = os.path.join(self._save_data_path, "test.csv")
        test_baseline_path = os.path.join(self._save_data_path, "test_baseline.csv")
        test_advanced_path = os.path.join(self._save_data_path, "test_advanced.csv")

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
            df_train_2 = self._last_event_features(df_train_2)
            print("feature engineering 2 ✅")

            features = [
                "period_time_seconds", "period", "x_coord", "y_coord",
                "distance_from_net", "shot_angle", "shot_type", "last_event_type",
                "last_event_x", "time_since_last_event", "last_event_distance",
                "rebound", "angle_change", "event_speed", "is_goal"
            ]
            df_train_2 = df_train_2[features].copy()

            print("saving advanced_train.csv ...")
            df_train_2.to_csv(advanced_train_path, index=False)
            print("saving advanced_train.csv ✅")
        else:
            print("advanced_train.csv already exists, skipping feature engineering 2")
            df_train_2 = pd.read_csv(advanced_train_path)

        # --- feature engineering for test data --- #
        if not os.path.exists(test_baseline_path) or not os.path.exists(test_advanced_path):
            print("feature engineering test data...")
            df_test_1 = self.feature_engineering_1(df_test)
            df_test_2 = self.feature_engineering_2(df_test)
            print("feature engineering test data ✅")

            print("saving feature engineered test data...")
            df_test_1.to_csv(test_baseline_path, index=False)
            df_test_2.to_csv(test_advanced_path, index=False)
            print("saving feature engineered test data ✅")
        else:
            print("Feature-engineered test CSVs already exist, skipping test feature engineering.")
                



def main():
    fe = FeatureEngineering()
    fe.create_data()

if __name__ == "__main__":
    main()
