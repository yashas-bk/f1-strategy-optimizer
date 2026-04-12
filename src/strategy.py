"""
strategy.py
F1 race strategy optimizer.

Enumerates all valid pit stop strategies (1-stop, 2-stop), simulates each
lap-by-lap using the LightGBM model, and returns strategies ranked by total
race time.

Works in two modes:
  Pre-race : current_lap=1, full race simulation from lap 1
  Mid-race : current_lap=N, reoptimises from lap N with current compound/tyre state

Key functions:
  optimise()       — find the top N strategies for a driver/race
  check_undercut() — check if pitting earlier than a rival is beneficial

Usage:
    from src.strategy import optimise, check_undercut, build_base_features
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import product as iproduct

from src.model import predict_stint, load_model
from src.pitstop_stats import get_pit_loss, load_pitstop_stats

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")

# F1 regulation: minimum dry-compound stint length (laps)
MIN_STINT = 8

COMPOUND_MAP = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Fallback max stint lengths when no historical data is available
DEFAULT_MAX_STINT = {"SOFT": 25, "MEDIUM": 33, "HARD": 42}


# ─────────────────────────────────────────
# TYRE PROFILE
# ─────────────────────────────────────────

def compute_tyre_profiles(years=(2023, 2024, 2025)) -> pd.DataFrame:
    """
    Computes per-track tyre profiles from historical race data:
      - Preferred starting compounds (dry races only, sorted by usage %)
      - Max stint length per compound (90th percentile of observed stints)

    Saves to data/processed/tyre_profiles.csv and returns the DataFrame.
    """
    frames = []
    for yr in years:
        path = os.path.join(RAW_DIR, f"race_{yr}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        raise FileNotFoundError("No race CSVs found.")
    df = pd.concat(frames, ignore_index=True)
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["Stint"]     = pd.to_numeric(df["Stint"],     errors="coerce")

    # ── Starting compounds (lap 1, stint 1, dry only) ─────────────────────
    starts = df[
        (df["LapNumber"] == 1) &
        (df["Stint"] == 1) &
        (df["Compound"].isin(DRY_COMPOUNDS))
    ].groupby(["EventName", "Compound"]).size().reset_index(name="Count")

    total_per_event = starts.groupby("EventName")["Count"].sum().reset_index(name="Total")
    starts = starts.merge(total_per_event, on="EventName")
    starts["StartPct"] = starts["Count"] / starts["Total"]

    # Keep compounds used by ≥15% of drivers — these are the realistic starting options
    start_compounds = (
        starts[starts["StartPct"] >= 0.15]
        .sort_values(["EventName", "StartPct"], ascending=[True, False])
        .groupby("EventName")["Compound"]
        .apply(list)
        .reset_index()
        .rename(columns={"Compound": "StartingCompounds"})
    )

    # ── Max stint lengths (90th pct, dry compounds only) ──────────────────
    stint_lens = (
        df[df["Compound"].isin(DRY_COMPOUNDS)]
        .groupby(["EventName", "Driver", "Year", "Stint", "Compound"])
        .size()
        .reset_index(name="StintLength")
    )
    stint_lens = stint_lens[stint_lens["StintLength"] >= 5]   # drop DNF/SC short stints

    max_stints = (
        stint_lens.groupby(["EventName", "Compound"])["StintLength"]
        .quantile(0.90)
        .reset_index()
        .rename(columns={"StintLength": "MaxStint"})
    )
    max_stints["MaxStint"] = max_stints["MaxStint"].round().astype(int)

    # Pivot to one row per event
    max_pivot = max_stints.pivot(index="EventName", columns="Compound", values="MaxStint")
    max_pivot.columns = [f"Max_{c}" for c in max_pivot.columns]
    max_pivot = max_pivot.reset_index()

    profiles = start_compounds.merge(max_pivot, on="EventName", how="outer")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    profiles.to_csv(os.path.join(PROCESSED_DIR, "tyre_profiles.csv"), index=False)
    print(f"Saved tyre_profiles.csv ({len(profiles)} tracks)")
    return profiles


def load_tyre_profiles() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "tyre_profiles.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("tyre_profiles.csv not found — run compute_tyre_profiles() first.")
    df = pd.read_csv(path)
    # StartingCompounds is stored as a string repr of a list — parse it back
    import ast
    df["StartingCompounds"] = df["StartingCompounds"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


def get_tyre_info(event_name: str, profiles: pd.DataFrame = None) -> dict:
    """
    Returns tyre profile for a specific track:
      starting_compounds : list of historically preferred starting compounds
      max_stint          : dict of {compound: max_laps}

    Falls back to sensible defaults if no data found.
    """
    if profiles is None:
        try:
            profiles = load_tyre_profiles()
        except FileNotFoundError:
            return {
                "starting_compounds": ["SOFT", "MEDIUM"],
                "max_stint": DEFAULT_MAX_STINT.copy(),
            }

    row = profiles[profiles["EventName"] == event_name]
    if row.empty:
        return {
            "starting_compounds": ["SOFT", "MEDIUM"],
            "max_stint": DEFAULT_MAX_STINT.copy(),
        }

    row = row.iloc[0]
    start_compounds = row["StartingCompounds"] if isinstance(row["StartingCompounds"], list) else ["SOFT", "MEDIUM"]

    max_stint = DEFAULT_MAX_STINT.copy()
    for compound in DRY_COMPOUNDS:
        col = f"Max_{compound}"
        if col in row and pd.notna(row[col]):
            max_stint[compound] = int(row[col])

    return {"starting_compounds": start_compounds, "max_stint": max_stint}


# ─────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────

@dataclass
class StrategyResult:
    """Holds the outcome of one simulated pit strategy."""
    compounds:      list            # e.g. ["SOFT", "MEDIUM"]
    pit_laps:       list            # e.g. [28] — lap on which driver enters pits
    lap_times:      pd.DataFrame    # per-lap: Lap, Compound, TyreLife, PredictedLapTime
    total_time:     float           # total race time in seconds
    total_time_str: str             # formatted "1:32:14.500"
    stops:          int             # number of pit stops

    def summary(self) -> str:
        compounds_str = " → ".join(self.compounds)
        pits_str      = ", ".join(f"L{p}" for p in self.pit_laps)
        return f"{compounds_str}  |  Pits: {pits_str}  |  Total: {self.total_time_str}"


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _format_time(total_seconds: float) -> str:
    """Converts seconds to H:MM:SS.mmm or M:SS.mmm string."""
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes}:{seconds:06.3f}"


def _compound_sequences(available: list, n_stints: int):
    """
    Yields all compound sequences of length n_stints where at least
    2 different compounds are used (F1 regulation).
    """
    for seq in iproduct(available, repeat=n_stints):
        if len(set(seq)) >= 2:
            yield list(seq)


def _pit_lap_combos(current_lap: int, total_laps: int, n_stops: int, step: int = 3):
    """
    Yields all valid pit lap tuples given minimum stint constraint.
    pit_laps[i] is the last lap of stint i (driver pits at the end of that lap).

    step: sample every Nth lap instead of every lap.
          step=1 → exhaustive (slow), step=3 → ~9x fewer combinations (fast).
          The optimal pit window rarely shifts by just 1 lap so step=3 is fine.
    """
    if n_stops == 1:
        lo = current_lap + MIN_STINT - 1
        hi = total_laps  - MIN_STINT
        for p in range(lo, hi + 1, step):
            yield (p,)

    elif n_stops == 2:
        lo1 = current_lap + MIN_STINT - 1
        hi1 = total_laps  - 2 * MIN_STINT
        for p1 in range(lo1, hi1 + 1, step):
            lo2 = p1 + MIN_STINT
            hi2 = total_laps - MIN_STINT
            for p2 in range(lo2, hi2 + 1, step):
                yield (p1, p2)


def _load_mappings():
    """Loads track and driver encoding mappings from JSON."""
    track_path  = os.path.join(PROCESSED_DIR, "track_mapping.json")
    driver_path = os.path.join(PROCESSED_DIR, "driver_mapping.json")
    track_map   = json.load(open(track_path))  if os.path.exists(track_path)  else {}
    driver_map  = json.load(open(driver_path)) if os.path.exists(driver_path) else {}
    return track_map, driver_map


# ─────────────────────────────────────────
# BASE FEATURES BUILDER
# ─────────────────────────────────────────

def build_base_features(
    driver:              str,
    event_name:          str,
    year:                int,   # noqa: ARG001 — kept for caller clarity
    fp2_long_run_pace:   float,
    fp2_deg_rate:        float,
    quali_position:      int,
    gap_to_pole:         float,
    quali_teammate_gap:  float,
    driver_skill_score:  float,
    rolling_season_form: float,
    track_character_score: float,
    team_performance_index: float,
    degradation_rate:    float,
    track_temp:          float       = 35.0,
    track_temp_min:      float       = 20.0,
    track_temp_max:      float       = 50.0,
) -> dict:
    """
    Builds the base feature dict for a driver/weekend.
    All features that stay constant across stints are set here.
    Compound-specific features (CompoundEncoded, TempCompoundInteraction)
    are updated per stint inside simulate_strategy().

    Args:
        driver              : Driver code e.g. "VER"
        event_name          : e.g. "British Grand Prix"
        year                : Race year
        fp2_long_run_pace   : Driver's average race-sim lap time from FP2 (seconds)
        fp2_deg_rate        : Tyre degradation rate from FP2 (s/lap)
        quali_position      : Grid position (1 = pole)
        gap_to_pole         : Gap to pole in qualifying (seconds)
        quali_teammate_gap  : Gap to teammate in qualifying (seconds)
        driver_skill_score  : Pre-computed driver skill score [0–1]
        rolling_season_form : Driver's rolling 3-race median pace (seconds)
        track_character_score: Driver's historical pace at this circuit type
        team_performance_index: Team's average relative pace this season segment
        degradation_rate    : Track+compound tyre deg rate (s/lap of tyre wear)
        track_temp          : Track temperature (°C)
        track_temp_min/max  : Min/max track temp for normalisation

    Returns:
        dict with all FEATURE_COLUMNS populated except compound-specific ones.
    """
    track_map, driver_map = _load_mappings()

    track_encoded  = track_map.get(event_name, -1)
    driver_encoded = driver_map.get(driver, -1)

    if track_temp_max > track_temp_min:
        track_temp_norm = (track_temp - track_temp_min) / (track_temp_max - track_temp_min)
    else:
        track_temp_norm = 0.5
    track_temp_norm = float(np.clip(track_temp_norm, 0.0, 1.0))

    return {
        # Encoding
        "TrackEncoded":           track_encoded,
        "DriverEncoded":          driver_encoded,
        # Weather
        "TrackTempNorm":          track_temp_norm,
        # Team
        "TeamPerformanceIndex":   team_performance_index,
        # Weekend session
        "FP2LongRunPace":         fp2_long_run_pace,
        "FP2DegRate":             fp2_deg_rate,
        "QualiPosition":          quali_position,
        "GapToPole":              gap_to_pole,
        "QualiTeammateGap":       quali_teammate_gap,
        # Driver
        "DriverSkillScore":       driver_skill_score,
        "RollingSeasonForm":      rolling_season_form,
        "TrackCharacterScore":    track_character_score,
        # Tyre (compound-specific — will be overridden per stint)
        "DegradationRate":        degradation_rate,
        "CompoundEncoded":        2,        # placeholder, overridden per stint
        "TempCompoundInteraction": track_temp_norm * 2,  # placeholder
        # Per-lap (overridden inside predict_stint)
        "TyreLife":               1,
        "LapNumber":              1,
    }


# ─────────────────────────────────────────
# STRATEGY SIMULATOR
# ─────────────────────────────────────────

def simulate_strategy(
    compound_sequence: list,
    pit_laps:          list,
    base_features:     dict,
    model,
    total_laps:        int,
    event_name:        str,
    pit_stats_df:      pd.DataFrame = None,
    condition:         str          = "Normal",
    current_lap:       int          = 1,
    current_tyre_life: int          = 1,
    deg_by_compound:   dict         = None,
) -> tuple:
    """
    Simulates a single pit strategy lap-by-lap.

    Args:
        compound_sequence : e.g. ["SOFT", "MEDIUM"] for a 1-stop
        pit_laps          : e.g. [28] — lap at end of which driver pits
        base_features     : dict from build_base_features()
        model             : trained LGBMRegressor
        total_laps        : total race laps
        event_name        : used to look up pit stop time
        pit_stats_df      : pre-loaded pitstop_stats DataFrame (loads from disk if None)
        condition         : "Normal", "SC", or "VSC"
        current_lap       : first lap to simulate (1 = pre-race, >1 = mid-race)
        current_tyre_life : tyre life at current_lap (only relevant when current_lap > 1)
        deg_by_compound   : optional dict {"SOFT": 0.08, "MEDIUM": 0.05, ...}
                            overrides base_features["DegradationRate"] per compound

    Returns:
        (lap_df, total_time):
            lap_df     — DataFrame [Lap, Compound, TyreLife, PredictedLapTime]
            total_time — total race time in seconds (laps + pit stops)
    """
    # Stint boundaries
    # pit_laps[i] = last lap of stint i, so stint i+1 starts on pit_laps[i]+1
    stint_starts = [current_lap]  + [pl + 1  for pl in pit_laps]
    stint_ends   = list(pit_laps) + [total_laps]

    all_laps   = []
    total_time = 0.0

    for i, (compound, start, end) in enumerate(
        zip(compound_sequence, stint_starts, stint_ends)
    ):
        stint_len       = end - start + 1
        start_tyre_life = current_tyre_life if i == 0 else 1

        # Update compound-specific features
        bf = base_features.copy()
        bf["CompoundEncoded"] = COMPOUND_MAP.get(compound, 2)
        bf["TempCompoundInteraction"] = (
            base_features["TrackTempNorm"] * bf["CompoundEncoded"]
        )
        if deg_by_compound and compound in deg_by_compound:
            bf["DegradationRate"] = deg_by_compound[compound]

        stint_df = predict_stint(
            bf,
            start_tyre_life=start_tyre_life,
            stint_length=stint_len,
            model=model,
            total_laps=total_laps,
            start_lap=start,
        )
        stint_df["Compound"] = compound
        all_laps.append(stint_df)
        total_time += stint_df["PredictedLapTime"].sum()

        # Add pit stop time loss at end of every stint except the last
        if i < len(compound_sequence) - 1:
            pit_loss = get_pit_loss(event_name, condition, pit_stats_df)
            total_time += pit_loss

    lap_df = pd.concat(all_laps, ignore_index=True)
    return lap_df, total_time


# ─────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────

def optimise(
    base_features:       dict,
    total_laps:          int,
    available_compounds: list,
    event_name:          str,
    model                = None,
    pit_stats_df:        pd.DataFrame = None,
    condition:           str          = "Normal",
    current_lap:         int          = 1,
    current_compound:    str          = None,
    current_tyre_life:   int          = 1,
    max_stops:           int          = 2,
    top_n:               int          = 10,
    deg_by_compound:     dict         = None,
    starting_compounds:  list         = None,
    max_stint_by_compound: dict       = None,
    step:                int          = 3,
) -> list:
    """
    Enumerates all valid pit strategies and returns the top N by total race time.

    Args:
        base_features       : dict from build_base_features()
        total_laps          : total race distance in laps
        available_compounds : list of compound names e.g. ["SOFT","MEDIUM","HARD"]
        event_name          : race name for pit loss lookup
        model               : trained model; loads from disk if None
        pit_stats_df        : pre-loaded pit stats; loads from disk if None
        condition           : "Normal", "SC", "VSC" — determines pit loss
        current_lap         : 1 for pre-race; >1 for mid-race reoptimisation
        current_compound    : fixed first compound when doing mid-race reopt
        current_tyre_life   : tyre age at current_lap (mid-race only)
        max_stops           : maximum number of pit stops to consider (1 or 2)
        top_n               : number of strategies to return
        deg_by_compound     : optional per-compound degradation override
        starting_compounds    : restrict which compounds can start the race.
                                e.g. ["MEDIUM"] if most drivers started on medium.
                                Derived from tyre_profiles.csv if None.
        max_stint_by_compound : hard cap on stint length per compound.
                                e.g. {"SOFT": 23, "MEDIUM": 32, "HARD": 42}
                                Prevents the model from suggesting 32-lap soft stints.
                                Derived from tyre_profiles.csv if None.
        step                  : sample pit windows every Nth lap (default 3).
                                step=1 is exhaustive but slow; step=3 gives ~9x speedup
                                with minimal accuracy loss.

    Returns:
        List of StrategyResult, sorted ascending by total_time (best first)
    """
    if model is None:
        model = load_model()
    if pit_stats_df is None:
        try:
            pit_stats_df = load_pitstop_stats()
        except FileNotFoundError:
            pit_stats_df = None

    dry = [c for c in available_compounds if c in DRY_COMPOUNDS]

    # Load tyre profile for this track to get data-driven defaults
    tyre_info = get_tyre_info(event_name)

    if starting_compounds is None:
        starting_compounds = [c for c in tyre_info["starting_compounds"] if c in dry]
        if not starting_compounds:
            starting_compounds = [c for c in ["SOFT", "MEDIUM"] if c in dry]
    else:
        starting_compounds = [c for c in starting_compounds if c in dry]

    if max_stint_by_compound is None:
        max_stint_by_compound = tyre_info["max_stint"]

    results = []

    for n_stops in range(1, max_stops + 1):
        n_stints = n_stops + 1

        for compound_seq in _compound_sequences(dry, n_stints):
            # Apply starting compound constraint (historically preferred compounds)
            if compound_seq[0] not in starting_compounds:
                continue

            # Mid-race: first compound is locked in (already on tyres)
            if current_lap > 1 and current_compound is not None:
                if compound_seq[0] != current_compound:
                    continue

            for pit_combo in _pit_lap_combos(current_lap, total_laps, n_stops, step=step):
                # Enforce max stint length per compound
                stint_starts = [current_lap] + [p + 1 for p in pit_combo]
                stint_ends   = list(pit_combo) + [total_laps]
                too_long = any(
                    (end - start + 1) > max_stint_by_compound.get(cmp, 999)
                    for cmp, start, end in zip(compound_seq, stint_starts, stint_ends)
                )
                if too_long:
                    continue

                try:
                    lap_df, total_time = simulate_strategy(
                        compound_sequence = compound_seq,
                        pit_laps          = list(pit_combo),
                        base_features     = base_features,
                        model             = model,
                        total_laps        = total_laps,
                        event_name        = event_name,
                        pit_stats_df      = pit_stats_df,
                        condition         = condition,
                        current_lap       = current_lap,
                        current_tyre_life = current_tyre_life,
                        deg_by_compound   = deg_by_compound,
                    )
                except Exception:
                    continue

                results.append(StrategyResult(
                    compounds      = compound_seq,
                    pit_laps       = list(pit_combo),
                    lap_times      = lap_df,
                    total_time     = total_time,
                    total_time_str = _format_time(total_time),
                    stops          = n_stops,
                ))

    results.sort(key=lambda r: r.total_time)
    return results[:top_n]


# ─────────────────────────────────────────
# UNDERCUT CHECKER
# ─────────────────────────────────────────

def check_undercut(
    your_base:            dict,
    rival_base:           dict,
    current_lap:          int,
    your_compound:        str,
    your_tyre_life:       int,
    rival_compound:       str,
    rival_tyre_life:      int,
    rival_planned_pit:    int,
    new_compound:         str,
    total_laps:           int,
    event_name:           str,
    gap_to_rival:         float = 0.0,
    model                       = None,
    pit_stats_df:         pd.DataFrame = None,
    window_laps:          int   = 5,
) -> dict:
    """
    Checks whether undercutting the rival is beneficial.

    Simulates both cars from current_lap for window_laps laps:
      - Your car: pits immediately (this lap), then runs new_compound
      - Rival car: stays out on rival_compound until rival_planned_pit, then pits

    Computes the gap between the two cars at the moment the rival exits the pits.
    A positive gap means you are ahead of the rival after the undercut.

    Args:
        your_base         : base features for your car
        rival_base        : base features for rival's car
        current_lap       : current race lap
        your_compound     : your current compound
        your_tyre_life    : your current tyre life
        rival_compound    : rival's current compound
        rival_tyre_life   : rival's current tyre life
        rival_planned_pit : lap on which rival is expected to pit
        new_compound      : compound you would fit after pitting
        total_laps        : total race laps
        event_name        : for pit stop time lookup
        gap_to_rival      : current gap between your car and rival (seconds)
                            positive = you are behind rival
        model             : trained model; loads from disk if None
        pit_stats_df      : pre-loaded pit stats
        window_laps       : how many laps after rival's pit exit to simulate

    Returns dict:
        undercut_possible   : True if you would be ahead after the undercut
        gap_at_rival_exit   : your gap to rival when rival exits pits
                              positive = you ahead, negative = rival still ahead
        your_time_to_exit   : total time you take from current_lap to rival's pit exit
        rival_time_to_exit  : total time rival takes over the same span
        lap_by_lap          : DataFrame comparing cumulative times lap by lap
        recommended_pit_lap : current_lap (pit now) or None if not beneficial
    """
    if model is None:
        model = load_model()
    if pit_stats_df is None:
        try:
            pit_stats_df = load_pitstop_stats()
        except FileNotFoundError:
            pit_stats_df = None

    pit_loss = get_pit_loss(event_name, "Normal", pit_stats_df)

    # ── Simulate your car: pit now ────────────────────────────────────────
    # Out-lap (current_lap): you pit at the end of current_lap
    # In-lap penalty: pit_loss added once
    # Then simulate on new_compound from current_lap+1

    rival_exit_lap = rival_planned_pit + 1      # first lap rival is back on track
    sim_end_lap    = rival_exit_lap + window_laps

    # Your car: 1 lap on current tyres (in-lap), then new compound
    your_bf = your_base.copy()
    your_bf["CompoundEncoded"] = COMPOUND_MAP.get(your_compound, 2)

    your_inlap = predict_stint(
        your_bf, your_tyre_life, 1, model, total_laps, current_lap
    )
    your_time = float(your_inlap["PredictedLapTime"].sum()) + pit_loss

    # Your car on new compound from current_lap+1 to sim_end_lap
    your_bf_new = your_base.copy()
    your_bf_new["CompoundEncoded"] = COMPOUND_MAP.get(new_compound, 2)
    your_bf_new["TempCompoundInteraction"] = (
        your_base["TrackTempNorm"] * your_bf_new["CompoundEncoded"]
    )
    new_stint_len = sim_end_lap - current_lap   # laps on new compound
    if new_stint_len > 0:
        your_new_stint = predict_stint(
            your_bf_new, 1, new_stint_len, model, total_laps, current_lap + 1
        )
        your_time += float(your_new_stint["PredictedLapTime"].sum())

    # ── Simulate rival car: stays out until rival_planned_pit ─────────────
    rival_bf = rival_base.copy()
    rival_bf["CompoundEncoded"] = COMPOUND_MAP.get(rival_compound, 2)
    rival_bf["TempCompoundInteraction"] = (
        rival_base["TrackTempNorm"] * rival_bf["CompoundEncoded"]
    )

    # Rival on current tyres: current_lap to rival_planned_pit inclusive
    rival_old_stint_len = rival_planned_pit - current_lap + 1
    rival_old = predict_stint(
        rival_bf, rival_tyre_life, rival_old_stint_len,
        model, total_laps, current_lap
    )
    rival_time = float(rival_old["PredictedLapTime"].sum()) + pit_loss

    # Rival on new tyres: rival_exit_lap to sim_end_lap
    rival_new_compound = new_compound  # assume rival fits same compound as you
    rival_bf_new = rival_base.copy()
    rival_bf_new["CompoundEncoded"] = COMPOUND_MAP.get(rival_new_compound, 2)
    rival_bf_new["TempCompoundInteraction"] = (
        rival_base["TrackTempNorm"] * rival_bf_new["CompoundEncoded"]
    )
    rival_new_len = window_laps
    if rival_new_len > 0:
        rival_new_stint = predict_stint(
            rival_bf_new, 1, rival_new_len, model, total_laps, rival_exit_lap
        )
        rival_time += float(rival_new_stint["PredictedLapTime"].sum())

    # ── Gap calculation ───────────────────────────────────────────────────
    # gap_to_rival: positive = you start behind rival
    # After simulation: positive your_time_delta = your car is slower (bad)
    your_time_delta  = your_time  - rival_time    # negative = you saved time
    gap_at_rival_exit = gap_to_rival - your_time_delta
    # positive gap_at_rival_exit = you are ahead of rival at their pit exit

    # ── Lap-by-lap comparison ─────────────────────────────────────────────
    your_laps = []
    # in-lap
    your_laps.append({
        "Lap": current_lap,
        "YourCompound": your_compound,
        "YourLapTime": float(your_inlap["PredictedLapTime"].iloc[0]) + pit_loss,
        "RivalCompound": rival_compound,
        "RivalLapTime": float(rival_old["PredictedLapTime"].iloc[0]),
    })
    # subsequent laps
    for j in range(1, new_stint_len):
        lap_num = current_lap + j
        rival_idx = j  # rival still on old tyres until rival_planned_pit
        your_lap_time   = float(your_new_stint["PredictedLapTime"].iloc[j - 1]) \
                          if j - 1 < len(your_new_stint) else np.nan
        rival_lap_time  = float(rival_old["PredictedLapTime"].iloc[rival_idx]) \
                          if rival_idx < len(rival_old) \
                          else (float(rival_new_stint["PredictedLapTime"].iloc[rival_idx - rival_old_stint_len])
                                if (rival_idx - rival_old_stint_len) < len(rival_new_stint)
                                else np.nan)
        your_laps.append({
            "Lap": lap_num,
            "YourCompound": new_compound,
            "YourLapTime": your_lap_time,
            "RivalCompound": rival_compound if lap_num <= rival_planned_pit else rival_new_compound,
            "RivalLapTime": rival_lap_time,
        })

    lap_by_lap = pd.DataFrame(your_laps)
    if not lap_by_lap.empty:
        lap_by_lap["GapDelta"] = (lap_by_lap["RivalLapTime"] - lap_by_lap["YourLapTime"]).cumsum()

    return {
        "undercut_possible":   gap_at_rival_exit > 0,
        "gap_at_rival_exit":   round(gap_at_rival_exit, 3),
        "your_time_to_exit":   round(your_time, 3),
        "rival_time_to_exit":  round(rival_time, 3),
        "lap_by_lap":          lap_by_lap,
        "recommended_pit_lap": current_lap if gap_at_rival_exit > 0 else None,
    }


# ─────────────────────────────────────────
# LOOKUP HELPERS
# ─────────────────────────────────────────

def get_driver_skill(driver: str) -> float:
    """Looks up a driver's skill score from driver_skill.csv."""
    path = os.path.join(PROCESSED_DIR, "driver_skill.csv")
    if not os.path.exists(path):
        return 0.5
    df = pd.read_csv(path)
    row = df[df["Driver"] == driver]
    return float(row["DriverSkillScore"].iloc[0]) if not row.empty else 0.5


def get_fp2_features(driver: str, year: int, round_number: int) -> dict:
    """Looks up FP2 long run pace and deg rate for a specific driver/race."""
    path = os.path.join(BASE_DIR, "data", "raw", f"fp2_profile_{year}.csv")
    if not os.path.exists(path):
        return {"FP2LongRunPace": np.nan, "FP2DegRate": np.nan}
    df  = pd.read_csv(path)
    row = df[(df["Driver"] == driver) & (df["RoundNumber"] == round_number)]
    if row.empty:
        return {"FP2LongRunPace": np.nan, "FP2DegRate": np.nan}
    return {
        "FP2LongRunPace": float(row["FP2AvgPace"].iloc[0]),
        "FP2DegRate":     float(row["FP2DegRate"].iloc[0]) if "FP2DegRate" in row else np.nan,
    }


def get_quali_features(driver: str, year: int, round_number: int) -> dict:
    """Looks up qualifying features for a specific driver/race."""
    path = os.path.join(BASE_DIR, "data", "raw", f"quali_{year}.csv")
    if not os.path.exists(path):
        return {"QualiPosition": np.nan, "GapToPole": np.nan, "QualiTeammateGap": np.nan}
    df  = pd.read_csv(path)
    row = df[(df["Driver"] == driver) & (df["RoundNumber"] == round_number)]
    if row.empty:
        return {"QualiPosition": np.nan, "GapToPole": np.nan, "QualiTeammateGap": np.nan}
    return {
        "QualiPosition":    float(row["QualiPosition"].iloc[0]),
        "GapToPole":        float(row["GapToPole"].iloc[0]) if "GapToPole" in row else np.nan,
        "QualiTeammateGap": float(row["QualiTeammateGap"].iloc[0]) if "QualiTeammateGap" in row else np.nan,
    }
