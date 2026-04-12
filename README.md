# F1 Strategy Optimizer

A machine learning–driven Formula 1 race strategy tool that predicts lap times and finds the optimal pit stop strategy for any driver at any race from the 2023–2025 seasons.

## Overview

The project uses FastF1 telemetry data to train a LightGBM model that predicts lap times relative to the event median. The strategy optimizer then enumerates all valid 1-stop and 2-stop strategies, simulates each lap-by-lap using the model, and returns the strategy with the lowest total race time.

A Streamlit dashboard exposes three tools:
- **Pre-Race Strategy Planner** — find the optimal strategy before the race starts
- **Live Race Reoptimizer** — reoptimize from the current lap mid-race
- **Undercut Analyser** — check whether pitting now to undercut a rival will work

## Project Structure

```
f1-strategy-optimizer/
├── app/
│   └── streamlit_app.py         # Streamlit dashboard (Phase 5)
├── data/
│   ├── raw/                     # FastF1 CSVs (race, quali, fp2, fp1, fp3)
│   └── processed/               # Feature-engineered data, model, mappings
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_pitstop_stats.ipynb
│   └── 05_strategy_optimizer.ipynb
├── src/
│   ├── data_loader.py           # FastF1 data extraction
│   ├── features.py              # Phase 2 feature engineering
│   ├── model.py                 # LightGBM training and prediction
│   ├── pitstop_stats.py         # Pit stop loss computation
│   └── strategy.py              # Strategy optimizer and undercut checker
└── README.md
```

## Model

**Algorithm:** LightGBM (leaf-wise gradient boosting)

**Target:** `RelativePace` — seconds above or below the event median lap time. Predicting relative pace instead of absolute lap time prevents year-over-year car development shifts from corrupting test set predictions.

**Absolute lap time reconstruction:**
```
AbsoluteLapTime = FP2LongRunPace + PredictedRelativePace
```

**Train/test split:** Time-based — 2023, 2024, and 2025 rounds 1–18 for training; 2025 rounds 19+ for testing.

**Results:** Train MAE 0.491s · Test MAE 0.520s on RelativePace

### Features (17)

| Category | Features |
|---|---|
| Tyre | TyreLife, CompoundEncoded, DegradationRate |
| Race context | LapNumber |
| Encoding | TrackEncoded, DriverEncoded |
| Weather | TrackTempNorm, TempCompoundInteraction |
| Team | TeamPerformanceIndex |
| Weekend | FP2LongRunPace, FP2DegRate, QualiPosition, GapToPole, QualiTeammateGap |
| Driver | DriverSkillScore, RollingSeasonForm, TrackCharacterScore |

**Driver skill score** is a composite of six components built from 2023–2024 data only (never current season): car-adjusted quali gap, teammate quali gap, car-adjusted race pace, tyre management, experience, and wet performance delta.

## Strategy Optimizer

The optimizer enumerates all valid 1-stop and 2-stop strategies subject to:
- F1 regulation: must use at least 2 different dry compounds
- Minimum stint length: 8 laps
- Per-track maximum stint lengths derived from historical data (e.g. SOFT ≤ 23 laps at Silverstone)
- Starting compound restricted to historically preferred compounds (e.g. MEDIUM at British GP)

Pit stop loss is computed from actual `PitOutTime − PitInTime` timestamps in the FastF1 data, broken down per track and per condition (Normal / Safety Car / VSC).

## Setup

```bash
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

Run notebooks in order (01 → 05) to reproduce the full pipeline from data extraction to strategy optimization.

## Data

Data is extracted via [FastF1](https://github.com/theOehrly/Fast-F1) for the 2023, 2024, and 2025 F1 seasons. Each session type (Race, Qualifying, FP1, FP2, FP3, Sprint) is stored as a separate CSV in `data/raw/`.

The raw data files are not included in this repository due to size. Run `notebooks/01_data_exploration.ipynb` to extract them.
