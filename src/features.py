"""
features.py
Reusable feature engineering functions for the F1 strategy optimizer.

Phase 2 feature set:
  Tyre       : TyreLife, CompoundEncoded, Compound_SOFT/MEDIUM/HARD, DegradationRate
  Race ctx   : LapNumber, FuelLoad, TrackEvolution, TrackEncoded, DriverEncoded
  Weather    : TrackTempNorm, TempCompoundInteraction
  Weekend    : FP2LongRunPace, FP2DegRate, QualiPosition, GapToPole, QualiTeammateGap
  Driver     : DriverSkillScore, RollingSeasonForm
  Track      : TrackCharacterScore
  Composite  : WeekendPaceScore  (dynamic weighted, see get_pace_weights)
"""

import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")


# ─────────────────────────────────────────
# BASIC LAP PARSING
# ─────────────────────────────────────────

def parse_laptimes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts timedelta string columns to seconds."""
    for col, new_col in [
        ("LapTime",      "LapTimeSeconds"),
        ("Sector1Time",  "Sector1Seconds"),
        ("Sector2Time",  "Sector2Seconds"),
        ("Sector3Time",  "Sector3Seconds"),
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
    for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        df[f"Compound_{c}"] = (df["Compound"] == c).astype(int)
    return df


def add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates tyre degradation rate per track per compound.

    Uses multiple regression: LapTimeSeconds ~ TyreLife + FuelLoad.
    Taking only the TyreLife coefficient isolates tyre wear from the
    fuel burn-off effect (which makes the car universally faster as
    the race progresses, regardless of tyre age).

    Requires at least 10 laps per group.
    """
    def _deg_coef(g):
        if len(g) < 10 or g["TyreLife"].nunique() < 2:
            return np.nan
        try:
            # Design matrix: [TyreLife, FuelLoad, intercept]
            X = np.column_stack([
                g["TyreLife"].values,
                g["FuelLoad"].values,
                np.ones(len(g)),
            ])
            y = g["LapTimeSeconds"].values
            coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return coefs[0]  # TyreLife coefficient only
        except np.linalg.LinAlgError:
            return np.nan

    deg_rate = (
        df.groupby(["EventName", "Compound"])
        .apply(_deg_coef, include_groups=False)
        .reset_index()
    )
    deg_rate.columns = ["EventName", "Compound", "DegradationRate"]
    return df.merge(deg_rate, on=["EventName", "Compound"], how="left")


def add_relative_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Adds session median and relative pace (seconds vs session median)."""
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
    """Adds FuelLoad (proxy) and TrackEvolution based on lap number."""
    race_length = (
        df.groupby(["Year", "EventName"])["LapNumber"]
        .max()
        .reset_index()
        .rename(columns={"LapNumber": "TotalLaps"})
    )
    df = df.merge(race_length, on=["Year", "EventName"], how="left")
    df["TrackEvolution"] = df["LapNumber"] / df["TotalLaps"]
    df["FuelLoad"]       = 1 - (df["LapNumber"] / df["TotalLaps"])
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalized track temp and compound-temp interaction."""
    if "TrackTemp" in df.columns and df["TrackTemp"].notna().any():
        t_min = df["TrackTemp"].min()
        t_max = df["TrackTemp"].max()
        if t_max > t_min:
            df["TrackTempNorm"] = (df["TrackTemp"] - t_min) / (t_max - t_min)
        else:
            df["TrackTempNorm"] = 0.5
        df["TempCompoundInteraction"] = df["TrackTempNorm"] * df["CompoundEncoded"]
    else:
        df["TrackTempNorm"]           = 0.5
        df["TempCompoundInteraction"] = 0.5 * df.get("CompoundEncoded", 2)
    return df


def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes track and driver as numeric IDs and saves mappings to JSON."""
    tracks      = sorted(df["EventName"].unique())
    track_map   = {t: i for i, t in enumerate(tracks)}
    df["TrackEncoded"] = df["EventName"].map(track_map)

    drivers     = sorted(df["Driver"].unique())
    driver_map  = {d: i for i, d in enumerate(drivers)}
    df["DriverEncoded"] = df["Driver"].map(driver_map)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "track_mapping.json"),  "w") as f:
        json.dump(track_map,  f, indent=2)
    with open(os.path.join(PROCESSED_DIR, "driver_mapping.json"), "w") as f:
        json.dump(driver_map, f, indent=2)

    return df


# ─────────────────────────────────────────
# TEAM PERFORMANCE INDEX (rolling segments)
# ─────────────────────────────────────────

def _season_segment(round_number: int) -> str:
    """Assigns a round to early / mid / late segment."""
    if round_number <= 7:
        return "early"
    elif round_number <= 15:
        return "mid"
    else:
        return "late"


def add_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    TeamPerformanceIndex — rolling per season segment (early/mid/late).

    Computes average RelativePace per team within each Year+Segment.
    Using segments (not full-season average) avoids concept drift masking:
    e.g. a team that develops rapidly mid-season won't have its early-season
    weakness hidden by later strong performances.
    """
    df["_Segment"] = df["RoundNumber"].apply(_season_segment)

    team_pace = (
        df.groupby(["Year", "_Segment", "Team"])["RelativePace"]
        .mean()
        .reset_index()
        .rename(columns={"RelativePace": "TeamPerformanceIndex"})
    )
    df = df.merge(team_pace, on=["Year", "_Segment", "Team"], how="left")
    df = df.drop(columns=["_Segment"])
    return df


# ─────────────────────────────────────────
# WEEKEND SESSION MERGES
# ─────────────────────────────────────────

def merge_fp2_features(race_df: pd.DataFrame,
                       fp2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges FP2 (or Sprint) long-run pace features onto race laps.

    Adds:
      FP2LongRunPace  — driver's avg race-sim lap time in FP2/Sprint
      FP2DegRate      — tyre degradation rate per lap in FP2/Sprint

    Joined on Driver + Year + RoundNumber (left join — NaN if FP2 missing).
    """
    fp2_cols = ["Driver", "Year", "RoundNumber", "FP2AvgPace", "FP2DegRate"]
    available = [c for c in fp2_cols if c in fp2_df.columns]
    fp2_sub = fp2_df[available].copy()

    rename = {"FP2AvgPace": "FP2LongRunPace"}
    fp2_sub = fp2_sub.rename(columns=rename)

    return race_df.merge(fp2_sub, on=["Driver", "Year", "RoundNumber"], how="left")


def merge_quali_features(race_df: pd.DataFrame,
                         quali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges qualifying features onto race laps.

    Adds:
      QualiPosition      — grid position
      GapToPole          — seconds behind pole (absolute pace signal)
      QualiTeammateGap   — gap to teammate (driver skill signal, car-neutral)

    Joined on Driver + Year + RoundNumber (left join).
    """
    quali_cols = ["Driver", "Year", "RoundNumber",
                  "QualiPosition", "GapToPole", "QualiTeammateGap"]
    available  = [c for c in quali_cols if c in quali_df.columns]
    quali_sub  = quali_df[available].copy()

    # Cap teammate gap at ±1.5s — removes anomalous sessions (crashes, red flags,
    # mechanical DNFs in quali) where one teammate posts an unrepresentative time.
    if "QualiTeammateGap" in quali_sub.columns:
        quali_sub["QualiTeammateGap"] = quali_sub["QualiTeammateGap"].clip(-1.5, 1.5)

    return race_df.merge(quali_sub, on=["Driver", "Year", "RoundNumber"], how="left")


# ─────────────────────────────────────────
# DRIVER SKILL SCORE
# ─────────────────────────────────────────

def build_driver_skill(race_df: pd.DataFrame,
                       quali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a driver skill score from 2023 + 2024 data only.
    Never uses current-season data — prevents leakage into the test set.

    Composite (weights sum to 1.0):
      0.30  car-adjusted quali gap    (driver quali pace beyond what the car gives)
      0.20  teammate quali gap        (direct teammate comparison, most car-neutral signal)
      0.20  car-adjusted race pace    (driver race pace beyond what the car gives)
      0.15  tyre management           (driver FP2 deg rate vs team average)
      0.10  experience                (races completed — prevents low-sample drivers ranking high)
      0.05  wet performance delta     (relative pace in rain vs dry)

    Returns a DataFrame with columns [Driver, DriverSkillScore].
    Saves to data/processed/driver_skill.csv.
    """
    race_hist  = race_df[race_df["Year"].isin([2023, 2024])].copy()
    quali_hist = quali_df[quali_df["Year"].isin([2023, 2024])].copy()

    drivers = sorted(
        set(race_hist["Driver"].unique()) |
        set(quali_hist["Driver"].unique())
    )

    # ── 1. Car-adjusted quali gap ──────────────────────────────────────────
    # Subtract team median GapToPole per round so only the driver's contribution
    # remains. Lower (more negative) = driver extracts more than the car gives.
    if "GapToPole" in quali_hist.columns and "Team" in quali_hist.columns:
        team_median_q = (
            quali_hist.groupby(["Year", "RoundNumber", "Team"])["GapToPole"]
            .median()
            .reset_index()
            .rename(columns={"GapToPole": "TeamMedianGap"})
        )
        q = quali_hist.merge(team_median_q, on=["Year", "RoundNumber", "Team"], how="left")
        q["CarAdjQualiGap"] = q["GapToPole"] - q["TeamMedianGap"]
        car_adj_quali = q.groupby("Driver")["CarAdjQualiGap"].mean()
    else:
        car_adj_quali = pd.Series(0.0, index=drivers)

    # ── 2. Teammate quali gap ──────────────────────────────────────────────
    if "QualiTeammateGap" in quali_hist.columns:
        teammate_gap = quali_hist.groupby("Driver")["QualiTeammateGap"].mean()
    else:
        teammate_gap = pd.Series(0.0, index=drivers)

    # ── 3. Car-adjusted race pace ─────────────────────────────────────────
    # Same logic as quali gap but for race laps: subtract team median
    # RelativePace per round. Captures race pace contribution beyond the car.
    if "RelativePace" in race_hist.columns and "Team" in race_hist.columns:
        team_median_r = (
            race_hist.groupby(["Year", "RoundNumber", "Team"])["RelativePace"]
            .median()
            .reset_index()
            .rename(columns={"RelativePace": "TeamMedianPace"})
        )
        r = race_hist.merge(team_median_r, on=["Year", "RoundNumber", "Team"], how="left")
        r["CarAdjRacePace"] = r["RelativePace"] - r["TeamMedianPace"]
        car_adj_race = r.groupby("Driver")["CarAdjRacePace"].mean()
    else:
        car_adj_race = pd.Series(0.0, index=drivers)

    # ── 4. Tyre management (driver FP2 deg rate vs team average) ──────────
    fp2_files = [
        os.path.join(RAW_DIR, f"fp2_profile_{yr}.csv")
        for yr in [2023, 2024]
    ]
    fp2_parts = []
    for fp in fp2_files:
        if os.path.exists(fp):
            fp2_parts.append(pd.read_csv(fp))
    if fp2_parts:
        fp2_hist = pd.concat(fp2_parts, ignore_index=True)
        fp2_hist = fp2_hist[fp2_hist["Year"].isin([2023, 2024])]
        if "FP2DegRate" in fp2_hist.columns and "Team" in fp2_hist.columns:
            team_deg = (
                fp2_hist.groupby(["Year", "RoundNumber", "Team"])["FP2DegRate"]
                .mean()
                .reset_index()
                .rename(columns={"FP2DegRate": "TeamAvgDeg"})
            )
            fp2_hist = fp2_hist.merge(team_deg, on=["Year", "RoundNumber", "Team"], how="left")
            fp2_hist["DegVsTeam"] = fp2_hist["FP2DegRate"] - fp2_hist["TeamAvgDeg"]
            tyre_mgmt = fp2_hist.groupby("Driver")["DegVsTeam"].mean()
            # Negate: lower deg than team average = better tyre management
            tyre_mgmt = -tyre_mgmt
        else:
            tyre_mgmt = pd.Series(0.0, index=drivers)
    else:
        tyre_mgmt = pd.Series(0.0, index=drivers)

    # ── 5. Experience (races completed in 2023+2024) ───────────────────────
    # Counts unique rounds per driver. Normalised 0-1. Prevents drivers with
    # very few races from dominating just because their small sample is clean.
    experience = (
        race_hist.groupby("Driver")[["Year", "RoundNumber"]]
        .apply(lambda g: g.drop_duplicates().shape[0], include_groups=False)
    )

    # ── 6. Wet performance delta ───────────────────────────────────────────
    if "Rainfall" in race_hist.columns and "RelativePace" in race_hist.columns:
        wet      = race_hist[race_hist["Rainfall"] == True]
        dry      = race_hist[race_hist["Rainfall"] == False]
        wet_pace = wet.groupby("Driver")["RelativePace"].mean()
        dry_pace = dry.groupby("Driver")["RelativePace"].mean()
        wet_delta = wet_pace - dry_pace
    else:
        wet_delta = pd.Series(0.0, index=drivers)

    # ── Normalise each component to [0, 1] and combine ────────────────────
    def _norm(s: pd.Series) -> pd.Series:
        s = s.reindex(drivers).fillna(s.median())
        mn, mx = s.min(), s.max()
        if mx > mn:
            return (s - mn) / (mx - mn)
        return pd.Series(0.5, index=s.index)

    skill = (
        0.30 * _norm(-car_adj_quali) +   # lower (more negative) adj gap = better
        0.20 * _norm(-teammate_gap)  +   # lower teammate gap = better
        0.20 * _norm(-car_adj_race)  +   # lower (more negative) adj race pace = better
        0.15 * _norm(tyre_mgmt)      +   # already negated, higher = better
        0.10 * _norm(experience)     +   # more races = more reliable signal
        0.05 * _norm(-wet_delta)         # lower wet delta = better wet performer
    )

    skill_df = skill.reset_index()
    skill_df.columns = ["Driver", "DriverSkillScore"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    skill_df.to_csv(os.path.join(PROCESSED_DIR, "driver_skill.csv"), index=False)
    print(f"  Saved driver_skill.csv ({len(skill_df)} drivers)")
    return skill_df


def add_driver_skill(race_df: pd.DataFrame,
                     skill_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merges DriverSkillScore onto race laps.
    Loads from data/processed/driver_skill.csv if skill_df not provided.
    """
    if skill_df is None:
        path = os.path.join(PROCESSED_DIR, "driver_skill.csv")
        if not os.path.exists(path):
            print("  ⚠️  driver_skill.csv not found — run build_driver_skill first")
            race_df["DriverSkillScore"] = np.nan
            return race_df
        skill_df = pd.read_csv(path)
    return race_df.merge(skill_df[["Driver", "DriverSkillScore"]],
                         on="Driver", how="left")


# ─────────────────────────────────────────
# ROLLING SEASON FORM
# ─────────────────────────────────────────

def add_rolling_season_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    RollingSeasonForm — driver's average RelativePace over their last 3 races.

    Computed per driver, per year, ordered by RoundNumber.
    Uses shift(1) so the current race's data is never included (no leakage).
    Drivers with fewer than 3 prior races get NaN (filled with season median).
    """
    df = df.sort_values(["Year", "Driver", "RoundNumber", "LapNumber"])

    # Get one representative RelativePace per driver per round (median lap)
    round_pace = (
        df.groupby(["Year", "Driver", "RoundNumber"])["RelativePace"]
        .median()
        .reset_index()
        .rename(columns={"RelativePace": "RoundMedianPace"})
        .sort_values(["Year", "Driver", "RoundNumber"])
    )

    # Rolling mean of last 3 rounds, shifted by 1 to exclude current round
    round_pace["RollingSeasonForm"] = (
        round_pace.groupby(["Year", "Driver"])["RoundMedianPace"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df = df.merge(
        round_pace[["Year", "Driver", "RoundNumber", "RollingSeasonForm"]],
        on=["Year", "Driver", "RoundNumber"],
        how="left"
    )

    # Fill NaN (early-season drivers) with the season median
    season_median = df.groupby("Year")["RollingSeasonForm"].transform("median")
    df["RollingSeasonForm"] = df["RollingSeasonForm"].fillna(season_median)
    return df


# ─────────────────────────────────────────
# TRACK CHARACTER SCORE
# ─────────────────────────────────────────

def add_track_character_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    TrackCharacterScore — driver's mean RelativePace at each CircuitType
    (street / highspeed / technical / mixed), computed from all available
    history except the current round (shift by round to avoid leakage).

    A negative score = driver goes faster than median at this circuit type.
    """
    df = df.sort_values(["Driver", "CircuitType", "Year", "RoundNumber"])

    circuit_pace = (
        df.groupby(["Driver", "CircuitType", "Year", "RoundNumber"])["RelativePace"]
        .median()
        .reset_index()
        .rename(columns={"RelativePace": "RoundCircuitPace"})
        .sort_values(["Driver", "CircuitType", "Year", "RoundNumber"])
    )

    # Expanding mean up to (but not including) current round
    circuit_pace["TrackCharacterScore"] = (
        circuit_pace.groupby(["Driver", "CircuitType"])["RoundCircuitPace"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df = df.merge(
        circuit_pace[["Driver", "CircuitType", "Year", "RoundNumber",
                       "TrackCharacterScore"]],
        on=["Driver", "CircuitType", "Year", "RoundNumber"],
        how="left"
    )

    # Fill NaN (first appearance at this circuit type) with driver's overall median
    driver_median = df.groupby("Driver")["RelativePace"].transform("median")
    df["TrackCharacterScore"] = df["TrackCharacterScore"].fillna(driver_median)
    return df


# ─────────────────────────────────────────
# WEEKEND PACE SCORE (weighted composite)
# ─────────────────────────────────────────

PACE_WEIGHTS_EARLY = {          # Rounds 1-4: rolling form is sparse
    "fp2_long_run":      0.35,
    "quali_gap_to_pole": 0.20,
    "rolling_form":      0.18,
    "driver_skill":      0.10,
    "fp3_direction":     0.05,
    "track_character":   0.07,
    "prev_season":       0.04,
    "fp1":               0.01,
}

PACE_WEIGHTS_LATE = {           # Round 5+: enough rolling data to trust it
    "fp2_long_run":      0.25,
    "quali_gap_to_pole": 0.20,
    "rolling_form":      0.26,
    "driver_skill":      0.10,
    "fp3_direction":     0.05,
    "track_character":   0.07,
    "prev_season":       0.05,
    "fp1":               0.02,
}

SPRINT_WEIGHT_REMAP = {
    "fp2_long_run":  "sprint_result",
    "fp3_direction": "sprint_quali_result",
}


def get_pace_weights(round_number: int) -> dict:
    """Returns the WeekendPaceScore weight dict for the given round number."""
    return PACE_WEIGHTS_LATE if round_number >= 5 else PACE_WEIGHTS_EARLY


def add_weekend_pace_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    WeekendPaceScore — weighted composite of all pre-race pace signals.

    Components mapped to available columns:
      fp2_long_run      → FP2LongRunPace      (normalised per round)
      quali_gap_to_pole → GapToPole           (normalised per round)
      rolling_form      → RollingSeasonForm   (normalised per season)
      driver_skill      → DriverSkillScore    (already normalised 0-1)
      fp3_direction     → FP3BestPace         (if available, else 0)
      track_character   → TrackCharacterScore (normalised)
      prev_season       → not yet available → weight redistributed
      fp1               → FP1AvgPace         (if available, else 0)

    Normalisation is min-max per RoundNumber so scores are comparable
    across different circuit lengths.
    """
    def _round_norm(series: pd.Series, round_col: pd.Series) -> pd.Series:
        out = series.copy().astype(float)
        for rnd in round_col.unique():
            mask = round_col == rnd
            vals = out[mask]
            mn, mx = vals.min(), vals.max()
            if mx > mn:
                out[mask] = (vals - mn) / (mx - mn)
            else:
                out[mask] = 0.5
        return out

    COMPONENT_COL = {
        "fp2_long_run":      "FP2LongRunPace",
        "quali_gap_to_pole": "GapToPole",
        "rolling_form":      "RollingSeasonForm",
        "driver_skill":      "DriverSkillScore",
        "fp3_direction":     "FP3BestPace",
        "track_character":   "TrackCharacterScore",
        "fp1":               "FP1AvgPace",
    }

    # Normalise all available components
    normed = {}
    missing_weight = 0.0

    for key, col in COMPONENT_COL.items():
        if col in df.columns and df[col].notna().any():
            normed[key] = _round_norm(df[col].fillna(df[col].median()), df["RoundNumber"])
        else:
            missing_weight += 0  # handled per-round below

    # prev_season has no column yet — its weight will be redistributed
    SKIP_KEYS = {"prev_season"}

    scores = pd.Series(0.0, index=df.index)
    for rnd in df["RoundNumber"].unique():
        mask   = df["RoundNumber"] == rnd
        rnd_num = int(rnd)
        weights = get_pace_weights(rnd_num)

        # Identify which components are available for this round
        available_w  = {k: v for k, v in weights.items()
                        if k not in SKIP_KEYS and k in normed}
        unavailable_w = {k: v for k, v in weights.items()
                         if k in SKIP_KEYS or k not in normed}
        total_missing = sum(unavailable_w.values())

        if total_missing > 0 and available_w:
            # Redistribute missing weight proportionally to available components
            total_avail = sum(available_w.values())
            available_w = {k: v + v / total_avail * total_missing
                           for k, v in available_w.items()}

        rnd_score = pd.Series(0.0, index=df[mask].index)
        for k, w in available_w.items():
            rnd_score += w * normed[k][mask]

        scores[mask] = rnd_score

    df["WeekendPaceScore"] = scores
    return df


# ─────────────────────────────────────────
# MASTER BUILD FUNCTION
# ─────────────────────────────────────────

def build_features(race_df: pd.DataFrame,
                   fp2_df:  pd.DataFrame = None,
                   fp3_df:  pd.DataFrame = None,
                   fp1_df:  pd.DataFrame = None,
                   quali_df: pd.DataFrame = None,
                   skill_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Master function — runs all Phase 2 feature engineering steps in order.

    Args:
        race_df  : Concatenated race laps (all years).
        fp2_df   : Concatenated FP2/Sprint profiles (all years).
        fp3_df   : Concatenated FP3/SprintQuali profiles (all years).
        fp1_df   : Concatenated FP1 profiles (all years).
        quali_df : Concatenated qualifying results (all years).
        skill_df : Pre-built driver skill DataFrame. If None, loads from CSV.

    Returns:
        Model-ready DataFrame with all FEATURE_COLUMNS populated.
    """
    print("Building features...")

    # Step 1: basic lap parsing and encoding
    df = parse_laptimes(race_df.copy())
    df = encode_compounds(df)
    df = add_race_context(df)
    df = add_relative_pace(df)
    df = add_weather_features(df)
    df = add_degradation_rate(df)
    df = add_encodings(df)
    print("  ✓ Basic features done")

    # Step 2: team performance (rolling segments)
    df = add_team_performance(df)
    print("  ✓ TeamPerformanceIndex done")

    # Step 3: merge weekend session data
    if fp2_df is not None:
        df = merge_fp2_features(df, fp2_df)
        print("  ✓ FP2 features merged")
    else:
        df["FP2LongRunPace"] = np.nan
        df["FP2DegRate"]     = np.nan

    if quali_df is not None:
        df = merge_quali_features(df, quali_df)
        print("  ✓ Quali features merged")
    else:
        df["QualiPosition"]    = np.nan
        df["GapToPole"]        = np.nan
        df["QualiTeammateGap"] = np.nan

    # Attach FP3 best pace for WeekendPaceScore fp3_direction component
    if fp3_df is not None and "FP3BestPace" in fp3_df.columns:
        fp3_sub = fp3_df[["Driver", "Year", "RoundNumber", "FP3BestPace"]].copy()
        df = df.merge(fp3_sub, on=["Driver", "Year", "RoundNumber"], how="left")
        print("  ✓ FP3 features merged")
    else:
        df["FP3BestPace"] = np.nan

    # Attach FP1 avg pace
    if fp1_df is not None and "FP1AvgPace" in fp1_df.columns:
        fp1_sub = fp1_df[["Driver", "Year", "RoundNumber", "FP1AvgPace"]].copy()
        df = df.merge(fp1_sub, on=["Driver", "Year", "RoundNumber"], how="left")
        print("  ✓ FP1 features merged")
    else:
        df["FP1AvgPace"] = np.nan

    # Step 4: driver-level features
    df = add_driver_skill(df, skill_df)
    print("  ✓ DriverSkillScore merged")

    df = add_rolling_season_form(df)
    print("  ✓ RollingSeasonForm done")

    df = add_track_character_score(df)
    print("  ✓ TrackCharacterScore done")

    # Step 5: composite weekend pace score
    df = add_weekend_pace_score(df)
    print("  ✓ WeekendPaceScore done")

    print(f"✅ Feature engineering complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLUMNS = [
    # Tyre
    "TyreLife",
    "CompoundEncoded",
    "DegradationRate",
    # Race context — FuelLoad and TrackEvolution dropped (same signal as LapNumber)
    "LapNumber",
    # Encoding
    "TrackEncoded",
    "DriverEncoded",
    # Weather
    "TrackTempNorm",
    "TempCompoundInteraction",
    # Team
    "TeamPerformanceIndex",
    # Weekend session
    "FP2LongRunPace",
    "FP2DegRate",
    "QualiPosition",
    "GapToPole",
    "QualiTeammateGap",
    # Driver
    "DriverSkillScore",
    "RollingSeasonForm",
    "TrackCharacterScore",
    # WeekendPaceScore dropped — composite of features already present individually
]

TARGET_COLUMN = "RelativePace"
