"""
features.py
Reusable feature engineering functions for the F1 strategy optimizer.
"""

import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_laptimes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts timedelta string columns to seconds."""
    for col, new_col in [
        ("LapTime", "LapTimeSeconds"),
        ("Sector1Time", "Sector1Seconds"),
        ("Sector2Time", "Sector2Seconds"),
        ("Sector3Time", "Sector3Seconds")
    ]:
        if col in df.columns:
            df[new_col] = df[col].apply(
                lambda v: pd.to_timedelta(v).total_seconds()
                if pd.notna(v) else None
            )
    return df


def encode_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ordinal and one-hot encoded compound columns."""
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3,
                    "INTERMEDIATE": 4, "WET": 5}
    df["CompoundEncoded"] = df["Compound"].map(compound_map)
    dummies = pd.get_dummies(df["Compound"], prefix="Compound")
    df = pd.concat([df, dummies], axis=1)
    return df


def add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates tyre degradation rate per track per compound."""
    deg_rate = (
        df.groupby(["EventName", "Compound"])
        .apply(lambda g: np.polyfit(g["TyreLife"],
               g["LapTimeSeconds"], 1)[0] if len(g) > 10 else np.nan)
        .reset_index()
    )
    deg_rate.columns = ["EventName", "Compound", "DegradationRate"]
    return df.merge(deg_rate, on=["EventName", "Compound"], how="left")


def add_relative_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Adds session median and relative pace columns."""
    session_median = (
        df.groupby(["Year", "EventName"])["LapTimeSeconds"]
        .median()
        .reset_index()
        .rename(columns={"LapTimeSeconds": "SessionMedianLapTime"})
    )
    df = df.merge(session_median, on=["Year", "EventName"], how="left")
    df["RelativePace"] = df["LapTimeSeconds"] - df["SessionMedianLapTime"]
    return df


def add_race_context(df: pd.DataFrame) -> pd.DataFrame:
    """Adds fuel load, track evolution, and total laps."""
    race_length = (
        df.groupby(["Year", "EventName"])["LapNumber"]
        .max()
        .reset_index()
        .rename(columns={"LapNumber": "TotalLaps"})
    )
    df = df.merge(race_length, on=["Year", "EventName"], how="left")
    df["TrackEvolution"] = df["LapNumber"] / df["TotalLaps"]
    df["FuelLoad"] = 1 - (df["LapNumber"] / df["TotalLaps"])
    return df


def add_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Adds team performance index per season."""
    team_pace = (
        df.groupby(["Year", "Team"])["RelativePace"]
        .mean()
        .reset_index()
        .rename(columns={"RelativePace": "TeamPerformanceIndex"})
    )
    return df.merge(team_pace, on=["Year", "Team"], how="left")


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized weather and interaction features."""
    if "TrackTemp" in df.columns:
        df["TrackTempNorm"] = (
            (df["TrackTemp"] - df["TrackTemp"].min()) /
            (df["TrackTemp"].max() - df["TrackTemp"].min())
        )
        df["TempCompoundInteraction"] = (
            df["TrackTempNorm"] * df["CompoundEncoded"]
        )
    else:
        df["TrackTempNorm"] = 0
        df["TempCompoundInteraction"] = 0
    return df


def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes track and driver as numeric IDs and saves mappings."""
    tracks = sorted(df["EventName"].unique())
    track_map = {t: i for i, t in enumerate(tracks)}
    df["TrackEncoded"] = df["EventName"].map(track_map)

    drivers = sorted(df["Driver"].unique())
    driver_map = {d: i for i, d in enumerate(drivers)}
    df["DriverEncoded"] = df["Driver"].map(driver_map)

    # Save mappings for use in the app
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    with open(os.path.join(processed_dir, "track_mapping.json"), "w") as f:
        json.dump(track_map, f, indent=2)
    with open(os.path.join(processed_dir, "driver_mapping.json"), "w") as f:
        json.dump(driver_map, f, indent=2)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — runs all feature engineering steps in order.
    Call this with the cleaned laps dataframe to get a model-ready dataframe.
    """
    print("Building features...")
    df = parse_laptimes(df)
    df = encode_compounds(df)
    df = add_relative_pace(df)
    df = add_race_context(df)
    df = add_team_performance(df)
    df = add_weather_features(df)
    df = add_degradation_rate(df)
    df = add_encodings(df)
    print(f"✅ Done. Shape: {df.shape}")
    return df


FEATURE_COLUMNS = [
    "TyreLife", "CompoundEncoded",
    "Compound_SOFT", "Compound_MEDIUM", "Compound_HARD",
    "DegradationRate", "LapNumber", "FuelLoad", "TrackEvolution",
    "TeamPerformanceIndex", "DriverEncoded", "TrackEncoded",
    "TrackTempNorm", "TempCompoundInteraction"
]

TARGET_COLUMN = "LapTimeSeconds"

# ─────────────────────────────────────────
# WEEKEND PACE SCORE WEIGHTS
# ─────────────────────────────────────────
# Dynamic weighting for the WeekendPaceScore composite feature.
# Early season (rounds 1–4): rolling form is sparse so FP2/Sprint
#   gets a higher weight as the primary weekend-specific signal.
# Late season (round 5+): enough rolling data exists to trust it,
#   so weight shifts from FP2 to rolling form.
# FP3/SprintQuali is capped at 0.05 in both regimes because it
#   is a qualifying-setup session with low predictive value for
#   race pace specifically.
#
# Usage in Phase 2 feature engineering:
#   weights = get_pace_weights(round_number)
#   score   = sum(weights[k] * features[k] for k in weights)

PACE_WEIGHTS_EARLY = {          # Rounds 1–4
    "fp2_long_run":       0.35,
    "quali_gap_to_pole":  0.20,
    "rolling_form":       0.18,
    "driver_skill":       0.10,
    "fp3_direction":      0.05,
    "track_character":    0.07,
    "prev_season":        0.04,
    "fp1":                0.01,
}

PACE_WEIGHTS_LATE = {           # Round 5 onwards
    "fp2_long_run":       0.25,
    "quali_gap_to_pole":  0.20,
    "rolling_form":       0.26,
    "driver_skill":       0.10,
    "fp3_direction":      0.05,
    "track_character":    0.07,
    "prev_season":        0.05,
    "fp1":                0.02,
}

SPRINT_WEIGHT_REMAP = {
    # For sprint weekends, fp2_long_run maps to the Sprint race result
    # and fp3_direction maps to the Sprint Qualifying result.
    # Key names stay the same so the score computation is identical.
    "fp2_long_run":  "sprint_result",
    "fp3_direction": "sprint_quali_result",
}


def get_pace_weights(round_number: int) -> dict:
    """
    Returns the correct WeekendPaceScore weight dict for the given round.
    Transition point is round 5 — enough rolling form data exists from
    round 5 onwards to make it a reliable signal.
    """
    return PACE_WEIGHTS_LATE if round_number >= 5 else PACE_WEIGHTS_EARLY