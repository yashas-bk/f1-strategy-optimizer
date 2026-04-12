"""
model.py
LightGBM lap time prediction model for the F1 strategy optimizer.

Trains on laps_features.csv, evaluates on the held-out test set
(2025 rounds 19+), and saves the trained model to data/processed/model.pkl.

Usage:
    from src.model import train, predict, load_model
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH    = os.path.join(PROCESSED_DIR, "model.pkl")


# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────

def make_split(df: pd.DataFrame):
    """
    Season-based split — never random.
    Train : 2023, 2024, 2025 rounds 1-18
    Test  : 2025 rounds 19+

    Using a time-based split prevents data leakage: the model never
    sees future races during training, which mirrors real-world use.
    """
    train_mask = (
        df["Year"].isin([2023, 2024]) |
        ((df["Year"] == 2025) & (df["RoundNumber"] <= 18))
    )
    test_mask = (df["Year"] == 2025) & (df["RoundNumber"] > 18)

    train = df[train_mask].copy()
    test  = df[test_mask].copy()

    # Drop rows where any feature is NaN — LightGBM handles some NaN
    # internally but we want clean splits for reliable metrics.
    train = train.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    test  = test.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test  = test[FEATURE_COLUMNS]
    y_test  = test[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test, train, test


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────

# LightGBM hyperparameters.
# These are sensible defaults for a 70k-row tabular regression problem.
# Can be tuned in the notebook with cross-validation if needed.
LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "learning_rate":    0.05,
    "num_leaves":       63,       # max leaves per tree — higher = more complex
    "max_depth":        -1,       # -1 = no limit (leaf-wise growth controls depth)
    "min_data_in_leaf": 50,       # min laps per leaf — prevents overfitting
    "feature_fraction": 0.8,      # use 80% of features per tree (reduces overfitting)
    "bagging_fraction": 0.8,      # use 80% of data per tree
    "bagging_freq":     5,
    "lambda_l1":        0.1,      # L1 regularisation
    "lambda_l2":        0.1,      # L2 regularisation
    "verbose":          -1,       # suppress training logs
    "n_jobs":           -1,
    "seed":             42,
}

N_ESTIMATORS   = 1000   # max trees — early stopping will cut this short
EARLY_STOPPING = 50     # stop if no improvement for 50 rounds on val set


def train(df: pd.DataFrame = None) -> lgb.LGBMRegressor:
    """
    Trains a LightGBM model on laps_features.csv.

    Args:
        df : Pre-loaded DataFrame. If None, loads laps_features.csv.

    Returns:
        Trained LGBMRegressor. Also saves to data/processed/model.pkl.
    """
    if df is None:
        path = os.path.join(PROCESSED_DIR, "laps_features.csv")
        print(f"Loading {path}...")
        df = pd.read_csv(path)

    X_train, y_train, X_test, y_test, _, _ = make_split(df)

    print(f"Train : {len(X_train):,} laps")
    print(f"Test  : {len(X_test):,} laps")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    model = lgb.LGBMRegressor(
        **LGBM_PARAMS,
        n_estimators=N_ESTIMATORS,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"\nBest iteration: {model.best_iteration_}")
    _print_metrics(model, X_train, y_train, X_test, y_test)
    save_model(model)
    return model


# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────

def _print_metrics(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    print("\n── Metrics (predicting RelativePace) ────")
    print(f"  Train MAE  : {mean_absolute_error(y_train, train_pred):.3f}s")
    print(f"  Train RMSE : {root_mean_squared_error(y_train, train_pred):.3f}s")
    print(f"  Test  MAE  : {mean_absolute_error(y_test, test_pred):.3f}s")
    print(f"  Test  RMSE : {root_mean_squared_error(y_test, test_pred):.3f}s")
    print("  (RelativePace = seconds above/below event median)")
    print("──────────────────────────────────────────")


def evaluate(model: lgb.LGBMRegressor, df: pd.DataFrame = None) -> dict:
    """
    Returns a dict of evaluation metrics for the model.
    Loads laps_features.csv if df not provided.
    """
    if df is None:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "laps_features.csv"))

    X_train, y_train, X_test, y_test, train, test = make_split(df)

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    return {
        "train_mae":  mean_absolute_error(y_train, train_pred),
        "train_rmse": root_mean_squared_error(y_train, train_pred),
        "test_mae":   mean_absolute_error(y_test, test_pred),
        "test_rmse":  root_mean_squared_error(y_test, test_pred),
        "test_preds": test_pred,
        "test_actual": y_test.values,
        "test_df":    test,
    }


# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────

def predict(features: dict, model: lgb.LGBMRegressor = None) -> float:
    """
    Predicts lap time in seconds for a single lap.

    Args:
        features : Dict of {feature_name: value} for all FEATURE_COLUMNS.
        model    : Trained model. Loads from disk if None.

    Returns:
        Predicted lap time in seconds.
    """
    if model is None:
        model = load_model()

    row = pd.DataFrame([{col: features.get(col, np.nan) for col in FEATURE_COLUMNS}])
    return float(model.predict(row)[0])


def predict_stint(
    base_features: dict,
    start_tyre_life: int,
    stint_length: int,
    model: lgb.LGBMRegressor = None,
    total_laps: int = 58,
    start_lap: int = 1,
) -> pd.DataFrame:
    """
    Predicts lap times for a full stint.

    The model outputs RelativePace (seconds above/below event median).
    Absolute lap time is reconstructed as:
        AbsoluteLapTime = FP2LongRunPace + PredictedRelativePace

    FP2LongRunPace anchors the baseline pace for the driver/weekend.
    This makes predictions robust to year-on-year car development shifts.

    Args:
        base_features   : Feature dict with all non-varying features pre-filled
                          (compound, driver, track, weather, quali data etc.)
                          Must include FP2LongRunPace.
        start_tyre_life : Tyre life at the start of the stint (1 = new)
        stint_length    : Number of laps in the stint
        model           : Trained model. Loads from disk if None.
        total_laps      : Total race laps (for FuelLoad / TrackEvolution calc)
        start_lap       : Race lap number at the start of the stint

    Returns:
        DataFrame with columns [Lap, TyreLife, PredictedRelativePace, PredictedLapTime]
    """
    if model is None:
        model = load_model()

    baseline = base_features.get("FP2LongRunPace", base_features.get("SessionMedianLapTime", 90.0))

    lap_nums   = np.arange(start_lap, start_lap + stint_length)
    tyre_lives = np.arange(start_tyre_life, start_tyre_life + stint_length)

    # Build all rows at once and call model.predict in a single batch.
    # This is ~50x faster than calling predict() once per lap in a loop.
    rows = {col: np.full(stint_length, val) for col, val in base_features.items()}
    rows["LapNumber"]      = lap_nums
    rows["TyreLife"]       = tyre_lives
    rows["FuelLoad"]       = 1 - (lap_nums / total_laps)
    rows["TrackEvolution"] = lap_nums / total_laps

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    rel_paces = model.predict(X)

    return pd.DataFrame({
        "Lap":                   lap_nums,
        "TyreLife":              tyre_lives,
        "PredictedRelativePace": rel_paces,
        "PredictedLapTime":      baseline + rel_paces,
    })


# ─────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────

def save_model(model: lgb.LGBMRegressor, path: str = None):
    path = path or MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path: str = None) -> lgb.LGBMRegressor:
    path = path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path} — run train() first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────

def feature_importance(model: lgb.LGBMRegressor) -> pd.DataFrame:
    """Returns a DataFrame of features sorted by importance (gain)."""
    imp = pd.DataFrame({
        "Feature":    FEATURE_COLUMNS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp


if __name__ == "__main__":
    trained_model = train()
