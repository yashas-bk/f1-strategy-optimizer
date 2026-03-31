
import fastf1
import pandas as pd
import os

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastf1.Cache.enable_cache(os.path.join(BASE_DIR, "data", "raw", "cache"))


def get_race_laps(year: int, round_number: int) -> pd.DataFrame:
    
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=True, messages=False)

        laps = session.laps.copy()

        # Add year and round info for tracking
        laps["Year"] = year
        laps["RoundNumber"] = round_number
        laps["EventName"] = session.event["EventName"]

        # Merge weather data onto laps using nearest timestamp
        weather = session.weather_data.copy()
        if not weather.empty:
            laps = pd.merge_asof(
                laps.sort_values("LapStartTime"),
                weather[["Time", "AirTemp", "TrackTemp", "Humidity",
                          "WindSpeed", "Rainfall"]].sort_values("Time"),
                left_on="LapStartTime",
                right_on="Time",
                direction="nearest"
            )

        return laps

    except Exception as e:
        print(f"  ⚠️  Could not load {year} Round {round_number}: {e}")
        return pd.DataFrame()


def collect_season(year: int, max_rounds: int = 25) -> pd.DataFrame:
    
    all_laps = []

    for round_num in range(1, max_rounds + 1):
        print(f"  Fetching {year} - Round {round_num}...", end=" ")
        laps = get_race_laps(year, round_num)

        if laps.empty:
            print("skipped.")
            continue

        all_laps.append(laps)
        print(f"✅ {len(laps)} laps loaded.")

    if not all_laps:
        print(f"No data collected for {year}.")
        return pd.DataFrame()

    return pd.concat(all_laps, ignore_index=True)


def save_season(df: pd.DataFrame, year: int):
    """Saves a season's lap data to CSV."""
    path = os.path.join(BASE_DIR, "data", "raw", f"season_{year}.csv")
    df.to_csv(path, index=False)
    print(f"\n💾 Saved {len(df)} laps to {path}")


def load_season(year: int) -> pd.DataFrame:
    """Loads a previously saved season CSV."""
    path = os.path.join(BASE_DIR, "data", "raw", f"season_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved data for {year}. Run collect_season() first.")
    return pd.read_csv(path)


if __name__ == "__main__":
    for year in [2023, 2024, 2025]:
        print(f"\n{'='*40}")
        print(f"Collecting {year} season...")
        print('='*40)
        df = collect_season(year)
        if not df.empty:
            save_season(df, year)