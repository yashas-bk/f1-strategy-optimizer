"""
pitstop_stats.py
Computes per-track average pit stop stationary time (in seconds) under three conditions:
  - Normal (green flag)
  - Safety Car (SC)
  - Virtual Safety Car (VSC)

Method:
  PitInTime  is recorded on the lap a driver enters the pits (end of that lap).
  PitOutTime is recorded on the NEXT lap (start of that lap).
  Pit stop time = PitOutTime(lap N+1) - PitInTime(lap N)

This is the actual time spent stationary in the pit box + pit lane entry/exit,
unaffected by how fast or slow the car was lapping under SC/VSC.

Output: data/processed/pitstop_stats.csv
"""

import os
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SC_CODE  = "4"
VSC_CODE = "6"


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert timedelta or timedelta-string series to float seconds."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _flag_condition(track_status: str) -> str:
    """Map a TrackStatus string to 'SC', 'VSC', or 'Normal'."""
    s = str(track_status)
    if SC_CODE in s:
        return "SC"
    if VSC_CODE in s:
        return "VSC"
    return "Normal"


def compute_pitstop_stats(years=(2023, 2024, 2025)) -> pd.DataFrame:
    """
    Loads race CSVs and computes two metrics per pit stop:

    1. PitStopTime  : raw time in pit lane = PitOutTime(lap N+1) - PitInTime(lap N)
       This is the same regardless of track condition.

    2. EffectiveLoss: time lost relative to competitors on track.
       Under SC/VSC, competitors go slowly so your net loss is less than
       the raw pit stop time.

       EffectiveLoss = PitStopTime × (NormalLapTime / ConditionLapTime)

       Where:
         NormalLapTime    = driver's median clean green-flag lap time in that race
         ConditionLapTime = median lap time of the 2 laps before the pit stop
                           (captures how slow the field is running under SC/VSC)

    Returns:
        stats    : DataFrame with per-track/condition aggregates
        pit_stops: DataFrame of every matched pit stop
    """
    frames = []
    for year in years:
        path = os.path.join(RAW_DIR, f"race_{year}.csv")
        if not os.path.exists(path):
            print(f"  Skipping {path} — not found")
            continue
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No race CSVs found in data/raw/")

    df = pd.concat(frames, ignore_index=True)

    df["PitInSec"]   = _to_seconds(df["PitInTime"])
    df["PitOutSec"]  = _to_seconds(df["PitOutTime"])
    df["LapTimeSec"] = _to_seconds(df["LapTime"])
    df["LapNumber"]  = pd.to_numeric(df["LapNumber"], errors="coerce")

    # ── Match PitInTime (lap N) to PitOutTime (lap N+1) ──────────────────
    pit_in = df[df["PitInSec"].notna()][
        ["Year", "RoundNumber", "EventName", "Driver", "LapNumber",
         "PitInSec", "TrackStatus"]
    ].copy()

    pit_out = df[df["PitOutSec"].notna()][
        ["Year", "RoundNumber", "Driver", "LapNumber", "PitOutSec"]
    ].copy()
    pit_out["LapNumber"] = pit_out["LapNumber"] - 1

    pit_stops = pit_in.merge(
        pit_out,
        on=["Year", "RoundNumber", "Driver", "LapNumber"],
        how="inner",
    )

    pit_stops["PitStopTime"] = pit_stops["PitOutSec"] - pit_stops["PitInSec"]
    pit_stops["Condition"]   = pit_stops["TrackStatus"].apply(_flag_condition)

    # Sanity filter
    pit_stops = pit_stops[
        (pit_stops["PitStopTime"] >= 1) &
        (pit_stops["PitStopTime"] <= 60)
    ].copy()

    df["Condition"] = df["TrackStatus"].astype(str).apply(_flag_condition)

    # ── Normal (green flag) median lap time per race (all drivers) ────────
    clean_mask = (
        df["LapTimeSec"].notna() &
        (df["LapNumber"] > 1) &
        (df["IsAccurate"] == True) &
        (df["Condition"] == "Normal")
    )
    normal_race_pace = (
        df[clean_mask]
        .groupby(["Year", "RoundNumber"])["LapTimeSec"]
        .median()
        .rename("NormalPace")
    )

    # ── SC / VSC median lap time per race ────────────────────────────────
    # Only use steady-state laps (prev lap also under same condition) to
    # exclude slow-down laps when SC/VSC is first deployed, which are faster
    # than true SC pace and would understate the pace ratio.
    df_sorted = df.sort_values(["Year", "RoundNumber", "Driver", "LapNumber"])
    df_sorted["PrevCondition"] = df_sorted.groupby(
        ["Year", "RoundNumber", "Driver"]
    )["Condition"].shift(1)

    sc_steady  = df_sorted[(df_sorted["Condition"] == "SC")  & (df_sorted["PrevCondition"] == "SC")  & df_sorted["LapTimeSec"].notna()]
    vsc_steady = df_sorted[(df_sorted["Condition"] == "VSC") & (df_sorted["PrevCondition"] == "VSC") & df_sorted["LapTimeSec"].notna()]

    sc_race_pace = (
        sc_steady["LapTimeSec"]
        .groupby([sc_steady["Year"], sc_steady["RoundNumber"]])
        .median()
        .rename("SCPace")
    )
    vsc_race_pace = (
        vsc_steady["LapTimeSec"]
        .groupby([vsc_steady["Year"], vsc_steady["RoundNumber"]])
        .median()
        .rename("VSCPace")
    )

    # ── Pace ratios: Normal / Condition — how much slower is SC/VSC ───────
    pace_ratios = pd.DataFrame({
        "NormalPace": normal_race_pace,
        "SCPace":     sc_race_pace,
        "VSCPace":    vsc_race_pace,
    }).reset_index()
    pace_ratios.columns = ["Year", "RoundNumber", "NormalPace", "SCPace", "VSCPace"]

    pace_ratios["SCRatio"]  = pace_ratios["NormalPace"] / pace_ratios["SCPace"]
    pace_ratios["VSCRatio"] = pace_ratios["NormalPace"] / pace_ratios["VSCPace"]

    pit_stops = pit_stops.merge(pace_ratios[["Year", "RoundNumber", "SCRatio", "VSCRatio"]],
                                on=["Year", "RoundNumber"], how="left")

    # EffectiveLoss = PitStopTime × (NormalPace / ConditionPace)
    # Normal: ratio = 1.0  →  EffectiveLoss = PitStopTime
    # SC:     ratio < 1.0  →  EffectiveLoss < PitStopTime (field is slow, you lose less)
    # VSC:    ratio < 1.0  →  EffectiveLoss < PitStopTime
    def _effective_loss(row):
        if row["Condition"] == "SC" and pd.notna(row.get("SCRatio")):
            return row["PitStopTime"] * row["SCRatio"]
        if row["Condition"] == "VSC" and pd.notna(row.get("VSCRatio")):
            return row["PitStopTime"] * row["VSCRatio"]
        return row["PitStopTime"]  # Normal: no adjustment

    pit_stops["EffectiveLoss"] = pit_stops.apply(_effective_loss, axis=1)
    pit_stops = pit_stops.dropna(subset=["PitStopTime", "EffectiveLoss"])

    # ── Aggregate per track + condition ───────────────────────────────────
    stats = (
        pit_stops.groupby(["EventName", "Condition"])
        .agg(
            MeanPitTime    =("PitStopTime",   "mean"),
            MedianPitTime  =("PitStopTime",   "median"),
            StdPitTime     =("PitStopTime",   "std"),
            MeanEffLoss    =("EffectiveLoss", "mean"),
            MedianEffLoss  =("EffectiveLoss", "median"),
            StdEffLoss     =("EffectiveLoss", "std"),
            Count          =("PitStopTime",   "count"),
        )
        .reset_index()
        .sort_values(["EventName", "Condition"])
    )

    return stats, pit_stops


def save_pitstop_stats(stats: pd.DataFrame, path: str = None):
    path = path or os.path.join(PROCESSED_DIR, "pitstop_stats.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stats.to_csv(path, index=False)
    print(f"Saved to {path}")


def load_pitstop_stats(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(PROCESSED_DIR, "pitstop_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No pitstop_stats.csv found — run compute_pitstop_stats() first.")
    return pd.read_csv(path)


def get_pit_loss(event_name: str, condition: str = "Normal",
                 stats_df: pd.DataFrame = None) -> float:
    """
    Returns the median EFFECTIVE pit loss in seconds for a given track and condition.
    EffectiveLoss = time lost relative to competitors on track (accounts for SC/VSC pace).
    Falls back to overall median for that condition if track has no data.
    """
    if stats_df is None:
        stats_df = load_pitstop_stats()

    row = stats_df[
        (stats_df["EventName"] == event_name) &
        (stats_df["Condition"] == condition)
    ]
    if not row.empty:
        return float(row["MedianEffLoss"].iloc[0])

    fallback = stats_df[stats_df["Condition"] == condition]["MedianEffLoss"]
    if not fallback.empty:
        return float(fallback.median())

    defaults = {"Normal": 22.0, "SC": 7.0, "VSC": 12.0}
    return defaults.get(condition, 22.0)


if __name__ == "__main__":
    print("Computing pit stop statistics...")
    stats, _ = compute_pitstop_stats()
    save_pitstop_stats(stats)
    print("\nPer-track pit stop times (seconds):")
    print(stats.to_string(index=False))
