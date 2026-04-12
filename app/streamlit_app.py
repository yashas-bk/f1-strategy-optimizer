"""
F1 Strategy Optimizer — Streamlit Dashboard
Phase 5

Three tabs:
  1. Pre-Race Strategy Planner  — find optimal strategy before the race
  2. Live Race Reoptimizer      — reoptimize mid-race from current state
  3. Undercut Analyser          — check if pitting now undercuts a rival

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.model import load_model
from src.pitstop_stats import load_pitstop_stats
from src.strategy import (
    build_base_features, optimise, check_undercut,
    get_driver_skill, get_fp2_features, get_quali_features,
    get_tyre_info, compute_tyre_profiles,
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

COMPOUND_COLOURS = {"SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#CACACA"}
DRY_COMPOUNDS    = ["SOFT", "MEDIUM", "HARD"]

TOTAL_LAPS_MAP = {
    "Abu Dhabi Grand Prix":        58,
    "Australian Grand Prix":       57,
    "Austrian Grand Prix":         71,
    "Azerbaijan Grand Prix":       51,
    "Bahrain Grand Prix":          57,
    "Belgian Grand Prix":          44,
    "British Grand Prix":          52,
    "Canadian Grand Prix":         70,
    "Chinese Grand Prix":          56,
    "Dutch Grand Prix":            72,
    "Emilia Romagna Grand Prix":   63,
    "Hungarian Grand Prix":        70,
    "Italian Grand Prix":          53,
    "Japanese Grand Prix":         53,
    "Las Vegas Grand Prix":        50,
    "Mexico City Grand Prix":      71,
    "Miami Grand Prix":            57,
    "Monaco Grand Prix":           78,
    "Qatar Grand Prix":            57,
    "Saudi Arabian Grand Prix":    50,
    "Singapore Grand Prix":        62,
    "Spanish Grand Prix":          66,
    "São Paulo Grand Prix":        71,
    "United States Grand Prix":    56,
}


# ─────────────────────────────────────────
# CACHED RESOURCE LOADING
# ─────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def _load_model():
    return load_model()

@st.cache_data(show_spinner="Loading pit stats...")
def _load_pit_stats():
    return load_pitstop_stats()

@st.cache_data(show_spinner="Loading tyre profiles...")
def _load_tyre_profiles():
    try:
        from src.strategy import load_tyre_profiles
        return load_tyre_profiles()
    except FileNotFoundError:
        return compute_tyre_profiles()

@st.cache_data(show_spinner="Loading race data...")
def _load_laps():
    return pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "data", "processed", "laps_features.csv")
    )


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _get_round(laps_df, event_name, year):
    row = laps_df[(laps_df["EventName"] == event_name) & (laps_df["Year"] == year)]
    if row.empty:
        return None
    return int(row["RoundNumber"].iloc[0])


def _get_context_features(laps_df, driver, event_name, year, round_number):
    """Looks up all features for a driver/race from stored CSVs + laps_features."""
    fp2   = get_fp2_features(driver, year, round_number)
    quali = get_quali_features(driver, year, round_number)
    quali["QualiTeammateGap"] = float(np.clip(quali.get("QualiTeammateGap", 0), -1.5, 1.5))
    skill = get_driver_skill(driver)

    race_laps = laps_df[
        (laps_df["Driver"] == driver) &
        (laps_df["Year"] == year) &
        (laps_df["RoundNumber"] == round_number)
    ]
    team_perf = float(race_laps["TeamPerformanceIndex"].median()) if not race_laps.empty else 0.0
    deg_rate  = float(race_laps["DegradationRate"].median())      if not race_laps.empty else 0.08
    rolling   = float(race_laps["RollingSeasonForm"].median())    if not race_laps.empty else 0.0
    tcs       = float(race_laps["TrackCharacterScore"].median())  if not race_laps.empty else 0.0

    return fp2, quali, skill, team_perf, deg_rate, rolling, tcs


def _lap_chart(result, driver, event_name, year):
    fig = go.Figure()
    for compound in result.compounds:
        sub = result.lap_times[result.lap_times["Compound"] == compound]
        fig.add_trace(go.Bar(
            x=sub["Lap"], y=sub["PredictedLapTime"],
            name=compound,
            marker_color=COMPOUND_COLOURS.get(compound, "steelblue"),
        ))
    for pit_lap in result.pit_laps:
        fig.add_vline(x=pit_lap + 0.5, line_dash="dash", line_color="white",
                      annotation_text=f"Pit L{pit_lap}",
                      annotation_font_color="white", annotation_font_size=11)
    fig.update_layout(
        title=f"{driver} — {event_name} {year} — {' → '.join(result.compounds)}",
        xaxis_title="Lap", yaxis_title="Lap Time (s)",
        barmode="stack",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white",
        legend_title="Compound", height=420,
    )
    return fig


def _deg_chart(result):
    fig = px.line(
        result.lap_times, x="TyreLife", y="PredictedLapTime", color="Compound",
        color_discrete_map=COMPOUND_COLOURS, markers=True,
        labels={"TyreLife": "Tyre Age (laps)", "PredictedLapTime": "Lap Time (s)"},
        height=350,
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white",
        title="Tyre Degradation Curve",
    )
    return fig


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="F1 Strategy Optimizer",
    page_icon="🏎️",
    layout="wide",
)

st.title("F1 Race Strategy Optimizer")
st.caption("Lap time predictions powered by LightGBM · Data: FastF1 2023–2025")

model         = _load_model()
pit_stats     = _load_pit_stats()
tyre_profiles = _load_tyre_profiles()
laps_df       = _load_laps()

all_events  = sorted(laps_df["EventName"].unique())
all_drivers = sorted(laps_df["Driver"].unique())
all_years   = sorted(laps_df["Year"].astype(int).unique(), reverse=True)

tab1, tab2, tab3 = st.tabs([
    "Pre-Race Strategy",
    "Live Reoptimizer",
    "Undercut Analyser",
])


# ═════════════════════════════════════════
# TAB 1 — PRE-RACE STRATEGY PLANNER
# ═════════════════════════════════════════

with tab1:
    st.subheader("Pre-Race Strategy Planner")
    st.markdown("Find the optimal pit strategy before the race starts.")

    c1, c2, c3 = st.columns(3)
    with c1:
        t1_driver = st.selectbox("Driver", all_drivers, index=all_drivers.index("VER"), key="t1_driver")
    with c2:
        t1_event  = st.selectbox("Race", all_events, index=all_events.index("British Grand Prix"), key="t1_event")
    with c3:
        t1_year   = st.selectbox("Year", all_years, key="t1_year")

    t1_round = _get_round(laps_df, t1_event, t1_year)
    t1_total_laps = TOTAL_LAPS_MAP.get(t1_event, 57)
    tyre_info = get_tyre_info(t1_event, tyre_profiles)

    c4, c5, c6 = st.columns(3)
    with c4:
        t1_track_temp = st.slider("Track Temp (°C)", 20, 60, 38, key="t1_temp")
    with c5:
        t1_max_stops = st.radio("Max pit stops", [1, 2], index=1, horizontal=True, key="t1_stops")
    with c6:
        t1_condition = st.selectbox("Race condition", ["Normal", "SC", "VSC"], key="t1_cond")

    # Show auto-detected tyre info
    with st.expander("Tyre profile (auto-detected from historical data)", expanded=False):
        st.write(f"**Preferred starting compounds:** {', '.join(tyre_info['starting_compounds'])}")
        st.write(f"**Max stint lengths:** {tyre_info['max_stint']}")
        st.caption("Override below if this race uses different compound allocations.")
        custom_starts = st.multiselect(
            "Starting compounds (override)",
            DRY_COMPOUNDS,
            default=tyre_info["starting_compounds"],
            key="t1_starts",
        )

    if t1_round is None:
        st.warning(f"No data found for {t1_event} {t1_year}.")
    else:
        fp2, quali, skill, team_perf, deg_rate, rolling, tcs = _get_context_features(
            laps_df, t1_driver, t1_event, t1_year, t1_round
        )

        with st.expander("Race weekend inputs (auto-filled · editable)", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                fp2_pace   = st.number_input("FP2 Long Run Pace (s)", value=round(fp2["FP2LongRunPace"], 3), step=0.001, format="%.3f", key="t1_fp2")
                fp2_deg    = st.number_input("FP2 Deg Rate (s/lap)",  value=round(fp2["FP2DegRate"], 4),     step=0.001, format="%.4f", key="t1_fp2deg")
                quali_pos  = st.number_input("Quali Position",         value=int(quali["QualiPosition"]),      step=1,     key="t1_qpos")
            with col_b:
                gap_pole   = st.number_input("Gap to Pole (s)",         value=round(quali["GapToPole"], 3),    step=0.001, format="%.3f", key="t1_gap")
                tm_gap     = st.number_input("Teammate Gap (s)",         value=round(quali["QualiTeammateGap"], 3), step=0.001, format="%.3f", key="t1_tmgap")
                deg_r      = st.number_input("Deg Rate (race)",          value=round(deg_rate, 4),              step=0.001, format="%.4f", key="t1_deg")

        run_btn = st.button("Run Optimizer", type="primary", key="t1_run")

        if run_btn:
            base = build_base_features(
                driver=t1_driver, event_name=t1_event, year=t1_year,
                fp2_long_run_pace=fp2_pace, fp2_deg_rate=fp2_deg,
                quali_position=quali_pos, gap_to_pole=gap_pole,
                quali_teammate_gap=tm_gap, driver_skill_score=skill,
                rolling_season_form=rolling, track_character_score=tcs,
                team_performance_index=team_perf, degradation_rate=deg_r,
                track_temp=t1_track_temp,
            )

            with st.spinner("Simulating strategies..."):
                strategies = optimise(
                    base_features=base, total_laps=t1_total_laps,
                    available_compounds=DRY_COMPOUNDS, event_name=t1_event,
                    model=model, pit_stats_df=pit_stats,
                    condition=t1_condition, max_stops=t1_max_stops, top_n=10,
                    starting_compounds=custom_starts if custom_starts else None,
                    step=3,
                )

            if not strategies:
                st.error("No valid strategies found. Check your inputs.")
            else:
                best = strategies[0]

                st.success(f"Best strategy: **{' → '.join(best.compounds)}** | Pit: {', '.join(f'L{p}' for p in best.pit_laps)} | Total: **{best.total_time_str}**")

                # Rankings table
                st.markdown("#### Top 10 Strategies")
                table_data = []
                for i, r in enumerate(strategies, 1):
                    table_data.append({
                        "Rank":       i,
                        "Strategy":   " → ".join(r.compounds),
                        "Pit Laps":   ", ".join(f"L{p}" for p in r.pit_laps),
                        "Stops":      r.stops,
                        "Total Time": r.total_time_str,
                        "vs Best (s)": f"+{r.total_time - best.total_time:.1f}" if i > 1 else "—",
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

                # Lap chart
                st.markdown("#### Lap-by-Lap Breakdown — Best Strategy")
                st.plotly_chart(_lap_chart(best, t1_driver, t1_event, t1_year), use_container_width=True)

                # Deg curve
                st.plotly_chart(_deg_chart(best), use_container_width=True)

                # Full lap table toggle
                with st.expander("Full lap-by-lap times"):
                    st.dataframe(
                        best.lap_times[["Lap", "Compound", "TyreLife", "PredictedLapTime"]]
                        .rename(columns={"PredictedLapTime": "Lap Time (s)"})
                        .round(3),
                        use_container_width=True, hide_index=True,
                    )


# ═════════════════════════════════════════
# TAB 2 — LIVE RACE REOPTIMIZER
# ═════════════════════════════════════════

with tab2:
    st.subheader("Live Race Reoptimizer")
    st.markdown("Enter the current race state to reoptimize strategy from this lap onwards.")

    c1, c2, c3 = st.columns(3)
    with c1:
        t2_driver = st.selectbox("Driver", all_drivers, index=all_drivers.index("VER"), key="t2_driver")
    with c2:
        t2_event  = st.selectbox("Race", all_events, index=all_events.index("British Grand Prix"), key="t2_event")
    with c3:
        t2_year   = st.selectbox("Year", all_years, key="t2_year")

    t2_round      = _get_round(laps_df, t2_event, t2_year)
    t2_total_laps = TOTAL_LAPS_MAP.get(t2_event, 57)
    t2_tyre_info  = get_tyre_info(t2_event, tyre_profiles)

    st.markdown("**Current Race State**")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        t2_current_lap  = st.number_input("Current Lap",    min_value=1, max_value=t2_total_laps - 1, value=20, key="t2_lap")
    with c5:
        t2_compound     = st.selectbox("Current Compound", DRY_COMPOUNDS, index=1, key="t2_compound")
    with c6:
        t2_tyre_life    = st.number_input("Tyre Age (laps)", min_value=1, max_value=40, value=15, key="t2_tl")
    with c7:
        t2_track_temp   = st.slider("Track Temp (°C)", 20, 60, 38, key="t2_temp")

    t2_condition = st.selectbox("Current track condition", ["Normal", "SC", "VSC"], key="t2_cond")

    if t2_round is None:
        st.warning(f"No data found for {t2_event} {t2_year}.")
    else:
        fp2, quali, skill, team_perf, deg_rate, rolling, tcs = _get_context_features(
            laps_df, t2_driver, t2_event, t2_year, t2_round
        )

        reopt_btn = st.button("Reoptimize from Lap " + str(t2_current_lap), type="primary", key="t2_run")

        if reopt_btn:
            base = build_base_features(
                driver=t2_driver, event_name=t2_event, year=t2_year,
                fp2_long_run_pace=fp2["FP2LongRunPace"], fp2_deg_rate=fp2["FP2DegRate"],
                quali_position=int(quali["QualiPosition"]), gap_to_pole=quali["GapToPole"],
                quali_teammate_gap=quali["QualiTeammateGap"], driver_skill_score=skill,
                rolling_season_form=rolling, track_character_score=tcs,
                team_performance_index=team_perf, degradation_rate=deg_rate,
                track_temp=t2_track_temp,
            )

            with st.spinner("Reoptimizing..."):
                strategies = optimise(
                    base_features=base, total_laps=t2_total_laps,
                    available_compounds=DRY_COMPOUNDS, event_name=t2_event,
                    model=model, pit_stats_df=pit_stats,
                    condition=t2_condition,
                    current_lap=t2_current_lap,
                    current_compound=t2_compound,
                    current_tyre_life=t2_tyre_life,
                    max_stops=2, top_n=10,
                    step=3,
                )

            if not strategies:
                st.error("No valid strategies found. You may be too late to pit.")
            else:
                best = strategies[0]
                laps_remaining = t2_total_laps - t2_current_lap + 1

                st.success(
                    f"Best remaining strategy: **{' → '.join(best.compounds)}** | "
                    f"Pit: {', '.join(f'L{p}' for p in best.pit_laps) if best.pit_laps else 'No more stops'} | "
                    f"Remaining time: **{best.total_time_str}**"
                )
                st.caption(f"{laps_remaining} laps remaining · Currently on {t2_compound} tyre age {t2_tyre_life}")

                table_data = []
                for i, r in enumerate(strategies, 1):
                    table_data.append({
                        "Rank":           i,
                        "Strategy":       " → ".join(r.compounds),
                        "Pit Laps":       ", ".join(f"L{p}" for p in r.pit_laps) or "Stay out",
                        "Remaining Time": r.total_time_str,
                        "vs Best (s)":    f"+{r.total_time - best.total_time:.1f}" if i > 1 else "—",
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
                st.plotly_chart(_lap_chart(best, t2_driver, t2_event, t2_year), use_container_width=True)


# ═════════════════════════════════════════
# TAB 3 — UNDERCUT ANALYSER
# ═════════════════════════════════════════

with tab3:
    st.subheader("Undercut Analyser")
    st.markdown("Should you pit now to undercut a rival, or stay out and attempt an overcut?")

    c1, c2, c3 = st.columns(3)
    with c1:
        t3_event = st.selectbox("Race", all_events, index=all_events.index("British Grand Prix"), key="t3_event")
    with c2:
        t3_year  = st.selectbox("Year", all_years, key="t3_year")
    with c3:
        t3_track_temp = st.slider("Track Temp (°C)", 20, 60, 38, key="t3_temp")

    t3_round      = _get_round(laps_df, t3_event, t3_year)
    t3_total_laps = TOTAL_LAPS_MAP.get(t3_event, 57)

    st.markdown("---")
    col_you, col_rival = st.columns(2)

    with col_you:
        st.markdown("**Your Car**")
        t3_your_driver   = st.selectbox("Driver", all_drivers, index=all_drivers.index("VER"), key="t3_you")
        t3_current_lap   = st.number_input("Current Lap", min_value=1, max_value=t3_total_laps - 1, value=25, key="t3_lap")
        t3_your_compound = st.selectbox("Current Compound", DRY_COMPOUNDS, index=1, key="t3_ycomp")
        t3_your_tl       = st.number_input("Tyre Age (laps)", min_value=1, max_value=40, value=18, key="t3_ytl")
        t3_new_compound  = st.selectbox("Compound to fit", DRY_COMPOUNDS, index=2, key="t3_newcomp")
        t3_gap_to_rival  = st.number_input("Gap to rival (s, positive = you behind)", value=1.8, step=0.1, format="%.1f", key="t3_gap")

    with col_rival:
        st.markdown("**Rival Car**")
        t3_rival_driver    = st.selectbox("Rival Driver", all_drivers, index=all_drivers.index("NOR"), key="t3_rival")
        st.write("")  # spacing
        t3_rival_compound  = st.selectbox("Rival Compound", DRY_COMPOUNDS, index=1, key="t3_rcomp")
        t3_rival_tl        = st.number_input("Rival Tyre Age (laps)", min_value=1, max_value=40, value=20, key="t3_rtl")
        t3_rival_pit       = st.number_input("Rival's Expected Pit Lap", min_value=t3_current_lap + 1,
                                              max_value=t3_total_laps - 5, value=30, key="t3_rpit")

    if t3_round is None:
        st.warning(f"No data found for {t3_event} {t3_year}.")
    else:
        undercut_btn = st.button("Analyse Undercut", type="primary", key="t3_run")

        if undercut_btn:
            fp2_y, quali_y, skill_y, tp_y, deg_y, roll_y, tcs_y = _get_context_features(
                laps_df, t3_your_driver, t3_event, t3_year, t3_round
            )
            fp2_r, quali_r, skill_r, tp_r, deg_r, roll_r, tcs_r = _get_context_features(
                laps_df, t3_rival_driver, t3_event, t3_year, t3_round
            )

            your_base = build_base_features(
                driver=t3_your_driver, event_name=t3_event, year=t3_year,
                fp2_long_run_pace=fp2_y["FP2LongRunPace"], fp2_deg_rate=fp2_y["FP2DegRate"],
                quali_position=int(quali_y["QualiPosition"]), gap_to_pole=quali_y["GapToPole"],
                quali_teammate_gap=np.clip(quali_y["QualiTeammateGap"], -1.5, 1.5),
                driver_skill_score=skill_y, rolling_season_form=roll_y,
                track_character_score=tcs_y, team_performance_index=tp_y,
                degradation_rate=deg_y, track_temp=t3_track_temp,
            )
            rival_base = build_base_features(
                driver=t3_rival_driver, event_name=t3_event, year=t3_year,
                fp2_long_run_pace=fp2_r["FP2LongRunPace"], fp2_deg_rate=fp2_r["FP2DegRate"],
                quali_position=int(quali_r["QualiPosition"]), gap_to_pole=quali_r["GapToPole"],
                quali_teammate_gap=np.clip(quali_r["QualiTeammateGap"], -1.5, 1.5),
                driver_skill_score=skill_r, rolling_season_form=roll_r,
                track_character_score=tcs_r, team_performance_index=tp_r,
                degradation_rate=deg_r, track_temp=t3_track_temp,
            )

            with st.spinner("Simulating undercut..."):
                result = check_undercut(
                    your_base=your_base, rival_base=rival_base,
                    current_lap=t3_current_lap,
                    your_compound=t3_your_compound, your_tyre_life=t3_your_tl,
                    rival_compound=t3_rival_compound, rival_tyre_life=t3_rival_tl,
                    rival_planned_pit=t3_rival_pit,
                    new_compound=t3_new_compound,
                    total_laps=t3_total_laps, event_name=t3_event,
                    gap_to_rival=t3_gap_to_rival,
                    model=model, pit_stats_df=pit_stats,
                )

            if result["undercut_possible"]:
                st.success(f"**UNDERCUT RECOMMENDED** — Pit on lap {t3_current_lap}")
            else:
                st.warning("**Undercut unlikely to work** — Stay out or consider overcut")

            m1, m2, m3 = st.columns(3)
            m1.metric("Gap before", f"{t3_gap_to_rival:+.2f}s", help="Positive = you behind")
            m2.metric(
                "Projected gap at rival pit exit",
                f"{result['gap_at_rival_exit']:+.2f}s",
                delta=f"{result['gap_at_rival_exit'] - t3_gap_to_rival:+.2f}s",
                delta_color="normal",
                help="Positive = you ahead",
            )
            m3.metric(
                "Time saved vs rival",
                f"{result['rival_time_to_exit'] - result['your_time_to_exit']:+.2f}s",
                help="Positive = you gained time",
            )

            # Lap time comparison chart
            lbl = result["lap_by_lap"]
            if not lbl.empty:
                fig_uc = go.Figure()
                fig_uc.add_trace(go.Scatter(
                    x=lbl["Lap"], y=lbl["YourLapTime"],
                    name=f"{t3_your_driver} (pit now on {t3_new_compound})",
                    line=dict(color="#00bfff", width=2), mode="lines+markers",
                ))
                fig_uc.add_trace(go.Scatter(
                    x=lbl["Lap"], y=lbl["RivalLapTime"],
                    name=f"{t3_rival_driver} (stay out on {t3_rival_compound})",
                    line=dict(color="#ff7043", width=2), mode="lines+markers",
                ))
                fig_uc.add_vline(x=t3_current_lap + 0.5, line_dash="dash", line_color="#00bfff",
                                 annotation_text=f"{t3_your_driver} pits", annotation_font_color="#00bfff")
                fig_uc.add_vline(x=t3_rival_pit + 0.5, line_dash="dash", line_color="#ff7043",
                                 annotation_text=f"{t3_rival_driver} pits", annotation_font_color="#ff7043")
                fig_uc.update_layout(
                    title=f"Undercut: {t3_your_driver} vs {t3_rival_driver}",
                    xaxis_title="Lap", yaxis_title="Lap Time (s)",
                    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
                    font_color="white", height=400,
                )
                st.plotly_chart(fig_uc, use_container_width=True)

                # Cumulative gap chart
                fig_gap = go.Figure()
                fig_gap.add_trace(go.Scatter(
                    x=lbl["Lap"], y=lbl["GapDelta"],
                    name="Cumulative time gained",
                    fill="tozeroy", line=dict(color="#00e676"), mode="lines",
                ))
                fig_gap.add_hline(y=t3_gap_to_rival, line_dash="dot", line_color="white",
                                  annotation_text="Gap needed to get ahead",
                                  annotation_font_color="white")
                fig_gap.add_hline(y=0, line_color="grey")
                fig_gap.update_layout(
                    title="Cumulative Time Gained by Undercutting",
                    xaxis_title="Lap", yaxis_title="Seconds gained",
                    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
                    font_color="white", height=320,
                )
                st.plotly_chart(fig_gap, use_container_width=True)
