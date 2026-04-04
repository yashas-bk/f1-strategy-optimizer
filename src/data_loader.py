"""
data_loader.py
Multi-session F1 data loader.
Collects Race, FP1, FP2, FP3, Qualifying, and driver skill data
across full seasons using FastF1.

Session priority for prediction:
  FP2 (long run pace) > Qualifying > FP3 > FP1
  
Driver skill is built from previous season data only.

Changes from v1:
  - Ergast API replaced with Jolpica (Ergast shut down 2024)
  - Teammate merge no-op (.apply(lambda g: g)) removed
  - Loop variable 'df' renamed to 'metric_df' to avoid shadowing
  - Quali score split into:
      _quali_abs  : car-adjusted gap to pole  (absolute pace, car-neutral)
      _quali_rel  : gap to teammate           (relative, reduced weight)
  - CarAdjustedGapToPole computed in _collect_skill_season
"""

import fastf1
import pandas as pd
import numpy as np
import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastf1.Cache.enable_cache(os.path.join(BASE_DIR, "data", "raw", "cache"))

# ─────────────────────────────────────────
# CORE SESSION LOADER
# ─────────────────────────────────────────

def get_session_laps(year: int, round_number: int,
                     session: str) -> pd.DataFrame:
    """
    Fetches lap data for a specific session.

    Args:
        year         : Season year
        round_number : Race number in the season
        session      : 'FP1', 'FP2', 'FP3', 'Q', or 'R'

    Returns:
        DataFrame of laps with session metadata and weather merged in.
    """
    try:
        s = fastf1.get_session(year, round_number, session)
        s.load(telemetry=False, weather=True, messages=False)

        laps = s.laps.copy()
        if laps.empty:
            return pd.DataFrame()

        laps["Year"]        = year
        laps["RoundNumber"] = round_number
        laps["EventName"]   = s.event["EventName"]
        laps["SessionType"] = session
        laps["CircuitType"] = _get_circuit_type(s.event["EventName"])

        # Merge nearest weather reading onto each lap.
        # Drop rows with null LapStartTime first — merge_asof requires
        # non-null keys (affects some sprint sessions e.g. Qatar 2023 SS).
        weather = s.weather_data.copy()
        if not weather.empty:
            laps = laps.dropna(subset=["LapStartTime"])
            laps = pd.merge_asof(
                laps.sort_values("LapStartTime"),
                weather[["Time", "AirTemp", "TrackTemp",
                          "Humidity", "WindSpeed", "Rainfall"]]
                      .sort_values("Time"),
                left_on="LapStartTime",
                right_on="Time",
                direction="nearest"
            )

        return laps

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} {session}: {e}")
        return pd.DataFrame()


def _get_circuit_type(event_name: str) -> str:
    """
    Classifies a circuit into one of four types.
    Used for the track character match score in Phase 2.

    Types:
      street   — Monaco, Baku, Singapore, Las Vegas, Miami, Jeddah
      highspeed — Monza, Spa, Silverstone, Suzuka, Bahrain
      technical — Hungary, Barcelona, Zandvoort, Imola
      mixed    — everything else
    """
    name = event_name.lower()

    street    = ["monaco", "baku", "singapore", "las vegas",
                 "miami", "jeddah", "saudi"]
    highspeed = ["monza", "spa", "silverstone", "suzuka",
                 "bahrain", "azerbaijan"]
    technical = ["hungar", "barcelona", "spain", "zandvoort",
                 "imola", "emilia"]

    if any(k in name for k in street):
        return "street"
    if any(k in name for k in highspeed):
        return "highspeed"
    if any(k in name for k in technical):
        return "technical"
    return "mixed"


# ─────────────────────────────────────────
# QUALIFYING EXTRACTOR
# ─────────────────────────────────────────

def get_qualifying_results(year: int,
                           round_number: int) -> pd.DataFrame:
    """
    Extracts qualifying performance per driver.

    Returns per driver:
      - Best qualifying lap time in seconds
      - Gap to pole position in seconds
      - Final qualifying position
      - Gap to teammate in seconds (pure driver vs car signal)
    """
    try:
        s = fastf1.get_session(year, round_number, "Q")
        s.load(telemetry=False, weather=False, messages=False)

        laps = s.laps.copy()
        if laps.empty:
            return pd.DataFrame()

        # Best lap per driver
        best = (
            laps.groupby("Driver")["LapTime"]
            .min()
            .reset_index()
            .rename(columns={"LapTime": "QualiTime"})
        )
        best["QualiTimeSeconds"] = best["QualiTime"].dt.total_seconds()

        # Gap to pole
        pole_time = best["QualiTimeSeconds"].min()
        best["GapToPole"] = best["QualiTimeSeconds"] - pole_time

        # Qualifying position
        best = best.sort_values("QualiTimeSeconds").reset_index(drop=True)
        best["QualiPosition"] = range(1, len(best) + 1)

        # Teammate gap — merge team info from laps
        driver_teams = laps[["Driver", "Team"]].drop_duplicates()
        best = best.merge(driver_teams, on="Driver", how="left")

        # For each driver, find their teammate's quali time
        # FIX: removed redundant .apply(lambda g: g) no-op
        teammate_times = (
            best[["Driver", "Team", "QualiTimeSeconds"]]
            .copy()
            .rename(columns={
                "Driver": "Teammate",
                "QualiTimeSeconds": "TeammateQualiTime"
            })
        )

        best = best.merge(teammate_times, on="Team", how="left")

        # Remove self-merge rows
        best = best[best["Driver"] != best["Teammate"]]

        best["QualiTeammateGap"] = (
            best["QualiTimeSeconds"] - best["TeammateQualiTime"]
        )

        # Keep one row per driver (in case of 3-car teams edge case)
        best = best.groupby("Driver").first().reset_index()

        best["Year"]        = year
        best["RoundNumber"] = round_number
        best["EventName"]   = s.event["EventName"]

        return best[["Driver", "Team", "Year", "RoundNumber",
                     "EventName", "QualiTimeSeconds", "GapToPole",
                     "QualiPosition", "QualiTeammateGap"]]

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} Qualifying: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# FP2 LONG RUN EXTRACTOR (PRIMARY PRACTICE SIGNAL)
# ─────────────────────────────────────────

def get_fp2_profile(year: int, round_number: int) -> pd.DataFrame:
    """
    Extracts FP2 long run pace profile per driver per compound.

    FP2 is the most important practice session for race prediction
    because teams run high fuel loads simulating race conditions.

    Filters out:
      - Outlaps and inlaps
      - Hot laps on low fuel (tyre life < 3)
      - Any lap over 150s or under 60s
    
    Returns per driver per compound:
      - Average pace on high fuel
      - Best pace
      - Observed degradation rate
      - Number of long run laps (confidence indicator)
    """
    try:
        laps = get_session_laps(year, round_number, "FP2")
        if laps.empty:
            return pd.DataFrame()

        laps["LapTimeSeconds"] = laps["LapTime"].apply(
            lambda v: pd.to_timedelta(v).total_seconds()
            if pd.notna(v) else None
        )

        # Filter to long run laps only
        clean = laps[
            laps["LapTimeSeconds"].notna() &
            (~laps["PitInTime"].notna()) &
            (~laps["PitOutTime"].notna()) &
            (laps["LapTimeSeconds"] < 150) &
            (laps["LapTimeSeconds"] > 60) &
            (laps["TyreLife"] >= 3) &
            (laps["Compound"].notna())
        ].copy()

        if clean.empty:
            return pd.DataFrame()

        # Stint-length filter: isolates high-fuel race sim laps only.
        # Teams switch to low-fuel quali sims at the end of FP2 — these
        # stints are typically 1–3 laps. A minimum of 6 laps per stint
        # removes those while keeping all genuine long run stints.
        if "Stint" in clean.columns:
            stint_lengths = (
                clean.groupby(["Driver", "Stint"])["LapNumber"]
                .count()
                .reset_index()
                .rename(columns={"LapNumber": "StintLength"})
            )
            clean = clean.merge(
                stint_lengths, on=["Driver", "Stint"], how="left")
            clean = clean[clean["StintLength"] >= 6]

        if clean.empty:
            return pd.DataFrame()

        def calc_deg_rate(g):
            if len(g) < 4:
                return np.nan
            slope, _ = np.polyfit(g["TyreLife"], g["LapTimeSeconds"], 1)
            return slope

        profile = (
            clean.groupby(["Driver", "Compound"])
            .apply(lambda g: pd.Series({
                "FP2AvgPace":    g["LapTimeSeconds"].mean(),
                "FP2BestPace":   g["LapTimeSeconds"].min(),
                "FP2DegRate":    calc_deg_rate(g),
                "FP2LapCount":   len(g),
                "FP2AvgTyreLife": g["TyreLife"].mean()
            }), include_groups=False)
            .reset_index()
        )

        profile["Year"]          = year
        profile["RoundNumber"]   = round_number
        profile["EventName"]     = clean["EventName"].iloc[0]
        profile["SessionSource"] = "FP2"

        return profile

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} FP2 profile: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# FP3 EXTRACTOR (SECONDARY PRACTICE SIGNAL)
# ─────────────────────────────────────────

def get_fp3_profile(year: int, round_number: int) -> pd.DataFrame:
    """
    Extracts FP3 pace profile per driver.

    FP3 is primarily for qualifying setup so it is a secondary
    signal. We extract average pace as a directional indicator
    only — this carries much less weight than FP2.
    """
    try:
        laps = get_session_laps(year, round_number, "FP3")
        if laps.empty:
            return pd.DataFrame()

        laps["LapTimeSeconds"] = laps["LapTime"].apply(
            lambda v: pd.to_timedelta(v).total_seconds()
            if pd.notna(v) else None
        )

        clean = laps[
            laps["LapTimeSeconds"].notna() &
            (~laps["PitInTime"].notna()) &
            (~laps["PitOutTime"].notna()) &
            (laps["LapTimeSeconds"] < 150) &
            (laps["LapTimeSeconds"] > 60) &
            (laps["TyreLife"] > 2)
        ].copy()

        if clean.empty:
            return pd.DataFrame()

        profile = (
            clean.groupby(["Driver"])
            .apply(lambda g: pd.Series({
                "FP3AvgPace":  g["LapTimeSeconds"].mean(),
                "FP3BestPace": g["LapTimeSeconds"].min(),
                "FP3LapCount": len(g)
            }), include_groups=False)
            .reset_index()
        )

        profile["Year"]        = year
        profile["RoundNumber"] = round_number
        profile["EventName"]   = clean["EventName"].iloc[0]

        return profile

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} FP3 profile: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# SPRINT EXTRACTORS (SPRINT WEEKENDS ONLY)
# ─────────────────────────────────────────

def get_sprint_profile(year: int, round_number: int) -> pd.DataFrame:
    """
    Extracts Sprint race pace profile per driver per compound.
    Used in place of FP2 for sprint weekends.

    The Sprint is a real race so all clean laps already represent
    competitive race conditions — no long-run filter needed.
    Outputs the same column schema as get_fp2_profile so all
    downstream code and CSV filenames stay unchanged.
    SessionSource column is set to 'Sprint' for traceability.
    """
    try:
        laps = get_session_laps(year, round_number, "S")
        if laps.empty:
            return pd.DataFrame()

        laps["LapTimeSeconds"] = laps["LapTime"].apply(
            lambda v: pd.to_timedelta(v).total_seconds()
            if pd.notna(v) else None
        )

        clean = laps[
            laps["LapTimeSeconds"].notna() &
            (~laps["PitInTime"].notna()) &
            (~laps["PitOutTime"].notna()) &
            (laps["LapTimeSeconds"] < 150) &
            (laps["LapTimeSeconds"] > 60) &
            (laps["Compound"].notna())
        ].copy()

        if clean.empty:
            return pd.DataFrame()

        def calc_deg_rate(g):
            if len(g) < 4:
                return np.nan
            slope, _ = np.polyfit(g["TyreLife"], g["LapTimeSeconds"], 1)
            return slope

        profile = (
            clean.groupby(["Driver", "Compound"])
            .apply(lambda g: pd.Series({
                "FP2AvgPace":     g["LapTimeSeconds"].mean(),
                "FP2BestPace":    g["LapTimeSeconds"].min(),
                "FP2DegRate":     calc_deg_rate(g),
                "FP2LapCount":    len(g),
                "FP2AvgTyreLife": g["TyreLife"].mean()
            }), include_groups=False)
            .reset_index()
        )

        profile["Year"]          = year
        profile["RoundNumber"]   = round_number
        profile["EventName"]     = clean["EventName"].iloc[0]
        profile["SessionSource"] = "Sprint"

        return profile

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} Sprint profile: {e}")
        return pd.DataFrame()


def get_sprint_quali_profile(year: int, round_number: int,
                             session_id: str = "SQ") -> pd.DataFrame:
    """
    Extracts Sprint Qualifying pace profile per driver.
    Used in place of FP3 for sprint weekends.

    Session identifier differs by era:
      'SS' — Sprint Shootout (2023, event_format='sprint_shootout')
      'SQ' — Sprint Qualifying (2024+, event_format='sprint_qualifying')
    The correct value is passed in from collect_full_weekend.

    Outputs the same column schema as get_fp3_profile so all
    downstream code and CSV filenames stay unchanged.
    SessionSource column is set to 'SprintQuali' for traceability.
    """
    try:
        laps = get_session_laps(year, round_number, session_id)
        if laps.empty:
            return pd.DataFrame()

        laps["LapTimeSeconds"] = laps["LapTime"].apply(
            lambda v: pd.to_timedelta(v).total_seconds()
            if pd.notna(v) else None
        )

        clean = laps[
            laps["LapTimeSeconds"].notna() &
            (~laps["PitInTime"].notna()) &
            (~laps["PitOutTime"].notna()) &
            (laps["LapTimeSeconds"] < 150) &
            (laps["LapTimeSeconds"] > 60)
        ].copy()

        if clean.empty:
            return pd.DataFrame()

        profile = (
            clean.groupby(["Driver"])
            .apply(lambda g: pd.Series({
                "FP3AvgPace":  g["LapTimeSeconds"].mean(),
                "FP3BestPace": g["LapTimeSeconds"].min(),
                "FP3LapCount": len(g)
            }), include_groups=False)
            .reset_index()
        )

        profile["Year"]          = year
        profile["RoundNumber"]   = round_number
        profile["EventName"]     = clean["EventName"].iloc[0]
        profile["SessionSource"] = "SprintQuali"

        return profile

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} Sprint Qualifying ({session_id}) profile: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# FP1 EXTRACTOR (LOWEST PRIORITY SIGNAL)
# ─────────────────────────────────────────

def get_fp1_profile(year: int, round_number: int) -> pd.DataFrame:
    """
    Extracts minimal FP1 data.
    FP1 is setup-focused and carries very little predictive
    weight. Only average pace is extracted as a fallback signal
    for weekends where FP2 data is incomplete.
    """
    try:
        laps = get_session_laps(year, round_number, "FP1")
        if laps.empty:
            return pd.DataFrame()

        laps["LapTimeSeconds"] = laps["LapTime"].apply(
            lambda v: pd.to_timedelta(v).total_seconds()
            if pd.notna(v) else None
        )

        clean = laps[
            laps["LapTimeSeconds"].notna() &
            (~laps["PitInTime"].notna()) &
            (~laps["PitOutTime"].notna()) &
            (laps["LapTimeSeconds"] < 150) &
            (laps["LapTimeSeconds"] > 60) &
            (laps["TyreLife"] > 3)
        ].copy()

        if clean.empty:
            return pd.DataFrame()

        profile = (
            clean.groupby(["Driver"])
            .apply(lambda g: pd.Series({
                "FP1AvgPace":  g["LapTimeSeconds"].mean(),
                "FP1LapCount": len(g)
            }), include_groups=False)
            .reset_index()
        )

        profile["Year"]        = year
        profile["RoundNumber"] = round_number
        profile["EventName"]   = clean["EventName"].iloc[0]

        return profile

    except Exception as e:
        print(f"    ⚠️  {year} R{round_number} FP1 profile: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────
# DRIVER SKILL BUILDER
# ─────────────────────────────────────────

def build_driver_skill_profile(years: list) -> pd.DataFrame:
    """
    Builds a driver skill score from historical data.
    Uses previous season(s) only — never current season —
    so it reflects stable driver ability independent of car.

    Metrics used (Tier 1 — pure skill):
      - Car-adjusted gap to pole  (absolute pace, car-neutral)
      - Qualifying gap to teammate (relative within-team delta)
      - Lap time consistency (std dev within stints)
      - Overtaking rate (positions gained per race)

    Metrics used (Tier 2 — performance under pressure):
      - Wet weather performance delta vs dry
      - Tyre management score (actual vs predicted degradation)

    Metrics used (Tier 3 — contextual):
      - Points per race normalized by team performance
        (rewards drivers scoring consistently in backmarker cars)

    Returns one row per driver with a composite DriverSkillScore
    between 0 (weakest) and 1 (strongest).
    """
    all_skill_data = []

    for year in years:
        print(f"    Building driver skill from {year}...")
        year_data = _collect_skill_season(year)
        if not year_data.empty:
            all_skill_data.append(year_data)

    if not all_skill_data:
        return pd.DataFrame()

    combined = pd.concat(all_skill_data, ignore_index=True)

    # Average across seasons for each driver
    skill = (
        combined.groupby("Driver")
        .agg(
            CarAdjustedGapToPole=("CarAdjustedGapToPole", "mean"),
            QualiGapToTeammate=("QualiGapToTeammate",   "mean"),
            LapConsistency=("LapConsistency",           "mean"),
            OvertakingRate=("OvertakingRate",           "mean"),
            WetDryDelta=("WetDryDelta",                 "mean"),
            TyreManagementScore=("TyreManagementScore", "mean"),
            NormalizedPointsPerRace=("NormalizedPointsPerRace", "mean")
        )
        .reset_index()
    )

    def normalize(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        norm = (series - mn) / (mx - mn)
        return 1 - norm if invert else norm

    # _quali_abs: car-adjusted gap to pole — lower = faster relative to
    # what the car should achieve, so invert (lower gap → higher score)
    skill["_quali_abs"]   = normalize(
        skill["CarAdjustedGapToPole"].abs(), invert=True)

    # _quali_rel: raw teammate gap — still useful but reduced weight
    # because closely matched teammates shouldn't be penalised
    skill["_quali_rel"]   = normalize(
        skill["QualiGapToTeammate"].abs(), invert=True)

    skill["_consistency"] = normalize(
        skill["LapConsistency"], invert=True)
    skill["_overtaking"]  = normalize(skill["OvertakingRate"])
    skill["_wet_dry"]     = normalize(
        skill["WetDryDelta"].abs(), invert=True)
    skill["_tyre_mgmt"]   = normalize(skill["TyreManagementScore"])
    skill["_points"]      = normalize(skill["NormalizedPointsPerRace"])

    # Weighted composite
    # _quali_abs gets the highest single weight because it is both
    # car-neutral and independent of teammate quality.
    # _quali_rel is kept at reduced weight as a secondary signal.
    skill["DriverSkillScore"] = (
        0.20 * skill["_quali_abs"]   +   # Tier 1 — absolute pace, car-neutral
        0.10 * skill["_quali_rel"]   +   # Tier 1 — teammate delta (reduced)
        0.20 * skill["_consistency"] +   # Tier 1
        0.20 * skill["_overtaking"]  +   # Tier 1
        0.15 * skill["_wet_dry"]     +   # Tier 2
        0.10 * skill["_tyre_mgmt"]   +   # Tier 2
        0.05 * skill["_points"]          # Tier 3
    )

    return skill[["Driver", "DriverSkillScore",
                  "CarAdjustedGapToPole", "QualiGapToTeammate",
                  "LapConsistency", "OvertakingRate",
                  "WetDryDelta", "TyreManagementScore",
                  "NormalizedPointsPerRace"]]


def _collect_skill_season(year: int) -> pd.DataFrame:
    """
    Collects raw skill metrics for all drivers in a season.
    Called internally by build_driver_skill_profile().
    """
    all_quali, all_race = [], []

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds   = schedule["RoundNumber"].dropna().astype(int).tolist()

    for rnd in rounds:
        q = get_qualifying_results(year, rnd)
        if not q.empty:
            all_quali.append(q)

        r = get_session_laps(year, rnd, "R")
        if not r.empty:
            r["LapTimeSeconds"] = r["LapTime"].apply(
                lambda v: pd.to_timedelta(v).total_seconds()
                if pd.notna(v) else None
            )
            all_race.append(r)

    if not all_quali or not all_race:
        return pd.DataFrame()

    quali_df = pd.concat(all_quali, ignore_index=True)
    race_df  = pd.concat(all_race,  ignore_index=True)

    # ── Metric 1a: Car-adjusted gap to pole
    # Subtracts each team's average gap to pole so that a backmarker
    # driver is judged against what their car can realistically achieve,
    # not against the pole setter. This removes car advantage entirely.
    team_avg_gap = (
        quali_df.groupby(["RoundNumber", "Team"])["GapToPole"]
        .transform("mean")
    )
    quali_df["CarAdjustedGapToPole"] = quali_df["GapToPole"] - team_avg_gap

    car_adj_gap = (
        quali_df.groupby("Driver")["CarAdjustedGapToPole"]
        .mean()
        .reset_index()
    )

    # ── Metric 1b: Average qualifying gap to teammate
    quali_gap = (
        quali_df.groupby("Driver")["QualiTeammateGap"]
        .mean()
        .reset_index()
        .rename(columns={"QualiTeammateGap": "QualiGapToTeammate"})
    )

    # ── Metric 2: Lap time consistency (std dev within stints)
    clean_race = race_df[
        race_df["LapTimeSeconds"].notna() &
        (race_df["LapTimeSeconds"] < 150) &
        (race_df["LapTimeSeconds"] > 60) &
        (~race_df["PitInTime"].notna()) &
        (~race_df["PitOutTime"].notna())
    ].copy()

    consistency = (
        clean_race.groupby(["Driver", "RoundNumber"])["LapTimeSeconds"]
        .std()
        .reset_index()
        .groupby("Driver")["LapTimeSeconds"]
        .mean()
        .reset_index()
        .rename(columns={"LapTimeSeconds": "LapConsistency"})
    )

    # ── Metric 3: Overtaking rate (positions gained per race)
    race_positions = race_df[
        race_df["LapNumber"] == race_df.groupby(
            ["RoundNumber", "Driver"])["LapNumber"].transform("max")
    ][["Driver", "RoundNumber", "GridPosition", "Position"]].drop_duplicates()

    if "GridPosition" in race_positions.columns:
        race_positions["PositionsGained"] = (
            race_positions["GridPosition"].astype(float) -
            race_positions["Position"].astype(float)
        )
        overtaking = (
            race_positions.groupby("Driver")["PositionsGained"]
            .mean()
            .reset_index()
            .rename(columns={"PositionsGained": "OvertakingRate"})
        )
    else:
        overtaking = pd.DataFrame(columns=["Driver", "OvertakingRate"])

    # ── Metric 4: Wet vs dry performance delta
    if "Rainfall" in race_df.columns:
        race_df["IsWet"] = race_df["Rainfall"].fillna(0) > 0
        wet_dry = (
            clean_race.groupby(["Driver", "IsWet"])["LapTimeSeconds"]
            .mean()
            .unstack(fill_value=np.nan)
            .reset_index()
        )
        if True in wet_dry.columns and False in wet_dry.columns:
            wet_dry["WetDryDelta"] = wet_dry[True] - wet_dry[False]
        else:
            wet_dry["WetDryDelta"] = 0
        wet_dry = wet_dry[["Driver", "WetDryDelta"]]
    else:
        wet_dry = pd.DataFrame(columns=["Driver", "WetDryDelta"])

    # ── Metric 5: Tyre management score
    def tyre_mgmt_score(g):
        if len(g) < 5:
            return np.nan
        slope, intercept = np.polyfit(g["TyreLife"], g["LapTimeSeconds"], 1)
        predicted = slope * g["TyreLife"] + intercept
        return float(np.mean(np.abs(g["LapTimeSeconds"] - predicted)))

    if "TyreLife" in clean_race.columns:
        tyre_mgmt = (
            clean_race.groupby(["Driver", "RoundNumber"])
            .apply(tyre_mgmt_score)
            .reset_index()
            .rename(columns={0: "TyreMgmt"})
            .groupby("Driver")["TyreMgmt"]
            .mean()
            .reset_index()
            .rename(columns={"TyreMgmt": "TyreManagementScore"})
        )
    else:
        tyre_mgmt = pd.DataFrame(
            columns=["Driver", "TyreManagementScore"])

    # ── Metric 6: Points per race normalized by team performance
    points_df = _fetch_points_per_race(year)

    # Merge all metrics together
    # FIX: renamed loop variable from 'df' to 'metric_df' to avoid
    # shadowing pandas DataFrame in notebook environments
    skill = car_adj_gap.merge(quali_gap, on="Driver", how="left")

    for metric_df, col in [
        (consistency, "LapConsistency"),
        (overtaking,  "OvertakingRate"),
        (wet_dry,     "WetDryDelta"),
        (tyre_mgmt,   "TyreManagementScore"),
        (points_df,   "NormalizedPointsPerRace")
    ]:
        if not metric_df.empty and col in metric_df.columns:
            skill = skill.merge(metric_df[["Driver", col]],
                                on="Driver", how="left")
        else:
            skill[col] = np.nan

    skill["Year"] = year
    return skill


def _fetch_points_per_race(year: int) -> pd.DataFrame:
    """
    Fetches driver standings from the Jolpica API (drop-in replacement
    for Ergast, which shut down in 2024) and normalizes points per race
    by team average.

    Normalization ensures a driver scoring 6pts per race in a
    backmarker car is rated higher than one scoring 6pts in a
    front-running car.
    """
    try:
        # FIX: Ergast is dead — replaced with Jolpica (identical JSON schema)
        url = f"https://api.jolpi.ca/ergast/f1/{year}/driverStandings/?limit=30"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        standings = (
            data["MRData"]["StandingsTable"]
               ["StandingsLists"][0]["DriverStandings"]
        )

        rows = []
        for s in standings:
            rows.append({
                "Driver":      s["Driver"]["code"],
                "Points":      float(s["points"]),
                "Wins":        int(s["wins"]),
                "Constructor": s["Constructors"][0]["name"]
            })

        df = pd.DataFrame(rows)

        sched_url = f"https://api.jolpi.ca/ergast/f1/{year}/races/?limit=30"
        sched     = requests.get(sched_url, timeout=10).json()
        num_races = len(sched["MRData"]["RaceTable"]["Races"])

        df["PointsPerRace"] = df["Points"] / max(num_races, 1)

        team_avg = (
            df.groupby("Constructor")["PointsPerRace"]
            .mean()
            .reset_index()
            .rename(columns={"PointsPerRace": "TeamAvgPoints"})
        )
        df = df.merge(team_avg, on="Constructor", how="left")

        df["NormalizedPointsPerRace"] = (
            df["PointsPerRace"] /
            df["TeamAvgPoints"].replace(0, np.nan)
        ).fillna(1.0)

        return df[["Driver", "NormalizedPointsPerRace", "PointsPerRace"]]

    except Exception as e:
        print(f"    ⚠️  Could not fetch points for {year}: {e}")
        return pd.DataFrame(columns=["Driver", "NormalizedPointsPerRace"])


# ─────────────────────────────────────────
# WEEKEND ORCHESTRATOR
# ─────────────────────────────────────────

def collect_full_weekend(year: int, round_number: int,
                         event_format: str = "conventional") -> dict:
    """
    Collects all session data for a single race weekend.

    For sprint weekends (event_format == 'sprint'):
      - fp2_profile comes from the Sprint race (S)
      - fp3_profile comes from Sprint Qualifying (SQ)
      FP1 and Qualifying sessions are the same for all weekend types.

    Returns a dict with keys:
      race, fp2_profile, fp3_profile, fp1_profile, quali
    """
    # FastF1 uses 'sprint_shootout' (2023) and 'sprint_qualifying' (2024+)
    # Sprint Qualifying session identifier also differs by era:
    #   'SS' = Sprint Shootout (2023), 'SQ' = Sprint Qualifying (2024+)
    is_sprint    = event_format in ("sprint_qualifying", "sprint_shootout")
    sq_id        = "SS" if event_format == "sprint_shootout" else "SQ"
    label        = "SPRINT" if is_sprint else "conventional"
    print(f"  Round {round_number} [{label}]...", end=" ")

    data = {
        "race": get_session_laps(year, round_number, "R"),
        "fp2_profile": (get_sprint_profile(year, round_number)
                        if is_sprint else
                        get_fp2_profile(year, round_number)),
        "fp3_profile": (get_sprint_quali_profile(year, round_number, sq_id)
                        if is_sprint else
                        get_fp3_profile(year, round_number)),
        "fp1_profile": get_fp1_profile(year, round_number),
        "quali":       get_qualifying_results(year, round_number),
    }

    race_laps = len(data["race"]) if not data["race"].empty else 0
    print(f"✅ {race_laps} race laps")
    return data


# ─────────────────────────────────────────
# SEASON COLLECTOR
# ─────────────────────────────────────────

def collect_season(year: int) -> dict:
    """
    Collects all data for a full season.

    Reads EventFormat from the FastF1 schedule to detect sprint weekends
    and passes it to collect_full_weekend so the correct session
    extractors are used per round.

    Returns a dict of combined DataFrames:
      race, fp2_profile, fp3_profile, fp1_profile, quali
    """
    buckets = {
        "race":        [],
        "fp2_profile": [],
        "fp3_profile": [],
        "fp1_profile": [],
        "quali":       [],
    }

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        rnd = event["RoundNumber"]
        if pd.isna(rnd):
            continue
        rnd          = int(rnd)
        event_format = str(event.get("EventFormat", "conventional")).lower()

        weekend = collect_full_weekend(year, rnd, event_format)
        for key in buckets:
            if not weekend[key].empty:
                buckets[key].append(weekend[key])

    return {
        key: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        for key, dfs in buckets.items()
    }


# ─────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────

def save_season(data: dict, year: int):
    """Saves all session DataFrames for a season to CSV."""
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    for session_type, df in data.items():
        if not df.empty:
            path = os.path.join(raw_dir, f"{session_type}_{year}.csv")
            df.to_csv(path, index=False)
            print(f"  💾 {session_type}_{year}.csv — {len(df):,} rows")


def load_season(year: int) -> dict:
    """Loads all saved session CSVs for a year."""
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    data    = {}
    keys    = ["race", "fp2_profile", "fp3_profile",
               "fp1_profile", "quali"]

    for key in keys:
        path = os.path.join(raw_dir, f"{key}_{year}.csv")
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
        else:
            print(f"  ⚠️  {key}_{year}.csv not found")
            data[key] = pd.DataFrame()

    return data


def load_driver_skill() -> pd.DataFrame:
    """Loads the saved driver skill profile."""
    path = os.path.join(BASE_DIR, "data", "processed",
                        "driver_skill.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "driver_skill.csv not found. "
            "Run build_driver_skill_profile() first."
        )
    return pd.read_csv(path)


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Sprint rounds only — non-sprint rounds already have correct data.
    # Collects only sprint weekend sessions, then merges into existing CSVs.
    SPRINT_ROUNDS = {
        2023: [17]
        
    }

    raw_dir = os.path.join(BASE_DIR, "data", "raw")

    for year, sprint_rnds in SPRINT_ROUNDS.items():
        print(f"\n{'='*45}")
        print(f"  {year} — collecting sprint rounds: {sprint_rnds}")
        print(f"{'='*45}")

        buckets = {k: [] for k in
                   ["race", "fp2_profile", "fp3_profile",
                    "fp1_profile", "quali"]}

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            rnd = event["RoundNumber"]
            if pd.isna(rnd) or int(rnd) not in sprint_rnds:
                continue
            rnd          = int(rnd)
            event_format = str(event.get(
                "EventFormat", "conventional")).lower()
            weekend = collect_full_weekend(year, rnd, event_format)
            for key in buckets:
                if not weekend[key].empty:
                    buckets[key].append(weekend[key])

        # Merge new sprint data into existing CSVs
        for key, frames in buckets.items():
            if not frames:
                continue
            new_data = pd.concat(frames, ignore_index=True)
            path     = os.path.join(raw_dir, f"{key}_{year}.csv")
            if os.path.exists(path):
                existing = pd.read_csv(path)
                # Drop any stale rows for these rounds before appending
                existing = existing[
                    ~existing["RoundNumber"].isin(sprint_rnds)]
                combined = (pd.concat([existing, new_data],
                                      ignore_index=True)
                              .sort_values("RoundNumber")
                              .reset_index(drop=True))
            else:
                combined = new_data
            combined.to_csv(path, index=False)
            print(f"  💾 {key}_{year}.csv — {len(combined):,} rows")

        print(f"  ✅ {year} sprint rounds complete")

    print(f"\n{'='*45}")
    print("  Building driver skill profile...")
    print(f"{'='*45}")
    skill_df = build_driver_skill_profile(years=[2023, 2024])

    if not skill_df.empty:
        skill_path = os.path.join(
            BASE_DIR, "data", "processed", "driver_skill.csv"
        )
        skill_df.to_csv(skill_path, index=False)
        print(f"  💾 driver_skill.csv — {len(skill_df)} drivers")
        print(skill_df[["Driver", "DriverSkillScore"]]
              .sort_values("DriverSkillScore", ascending=False)
              .to_string(index=False))