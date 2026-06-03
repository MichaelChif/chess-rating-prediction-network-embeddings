"""
train.py -- Train an XGBoost regression model to predict player Elo rating.

Inputs
------
data/processed/player_features.parquet   (output of features.py)

Outputs
-------
models/xgb_rating_model.joblib           trained XGBRegressor
outputs/metrics.json                     RMSE, MAE, R2 on train + test splits
outputs/predictions.csv                  per-player actuals, predictions, errors
outputs/feature_importance.png           horizontal bar chart (top-30 features)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import (
    MODEL_FILENAME,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_RANDOM_STATE,
    XGB_SUBSAMPLE,
)
from app.utils import ensure_dirs, get_logger, timer

logger = get_logger(__name__)

# Columns that are never model features
_EXCLUDE_COLS = {"player", "target_elo"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_features(df: pd.DataFrame) -> list[str]:
    """Return names of all numeric columns that are not excluded."""
    return [
        c for c in df.columns
        if c not in _EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _save_importance_plot(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path,
    top_n: int = 30,
) -> None:
    """Save a horizontal bar chart of the top-N features by importance."""
    fi = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values(ascending=True)   # reverse so largest bar is at top
    )

    n = len(fi)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.32)))

    bars = ax.barh(fi.index, fi.values, color="#2196F3", edgecolor="white", height=0.7)

    # Value labels at the end of each bar
    for bar, val in zip(bars, fi.values):
        ax.text(
            val + fi.values.max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=7.5,
        )

    ax.set_xlabel("Feature importance (gain)", fontsize=10)
    ax.set_title(
        f"XGBoost Feature Importance  (top {n})",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, fi.values.max() * 1.18)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance plot -> {out_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@timer
def train(
    features_df: Optional[pd.DataFrame] = None,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train an XGBoost regressor on *features_df* and persist all artefacts.

    Parameters
    ----------
    features_df : DataFrame, optional
        Output of ``extract_features`` (player_features.parquet).
        Loaded from disk when not supplied.

    Returns
    -------
    model : XGBRegressor
    metrics : dict
        Keys: ``train`` and ``test``, each a sub-dict with rmse/mae/r2.
        Also includes ``n_train``, ``n_test``, ``n_features``,
        ``best_iteration``.
    """
    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    if features_df is None:
        src = PROCESSED_DIR / "player_features.parquet"
        logger.info(f"Loading features from {src}")
        features_df = pd.read_parquet(src)

    feat_cols = _select_features(features_df)
    logger.info(
        f"Dataset: {len(features_df):,} players, "
        f"{len(feat_cols)} features"
    )

    X = features_df[feat_cols].astype(np.float32).values
    y = features_df["target_elo"].astype(np.float32).values

    # ------------------------------------------------------------------ #
    # 2. Train / test split                                                #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # Keep player names aligned with the same split for predictions.csv
    players = features_df["player"].values
    _, _, players_train, players_test = train_test_split(
        X, players,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info(f"Split: {len(X_train):,} train  |  {len(X_test):,} test")

    # ------------------------------------------------------------------ #
    # 3. Build and fit model                                               #
    # ------------------------------------------------------------------ #
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        random_state=XGB_RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        verbosity=0,
    )

    logger.info("Fitting XGBoost ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    best_iter = model.best_iteration
    logger.info(
        f"Training complete. "
        f"Best iteration: {best_iter} / {XGB_N_ESTIMATORS}"
    )

    # ------------------------------------------------------------------ #
    # 4. Evaluate                                                          #
    # ------------------------------------------------------------------ #
    train_metrics = _evaluate(y_train, model.predict(X_train))
    test_metrics  = _evaluate(y_test,  model.predict(X_test))

    logger.info("--- Train metrics ---")
    logger.info(f"  RMSE : {train_metrics['rmse']:.2f} Elo")
    logger.info(f"  MAE  : {train_metrics['mae']:.2f} Elo")
    logger.info(f"  R2   : {train_metrics['r2']:.4f}")
    logger.info("--- Test metrics  ---")
    logger.info(f"  RMSE : {test_metrics['rmse']:.2f} Elo")
    logger.info(f"  MAE  : {test_metrics['mae']:.2f} Elo")
    logger.info(f"  R2   : {test_metrics['r2']:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Persist artefacts                                                 #
    # ------------------------------------------------------------------ #
    ensure_dirs(MODELS_DIR, OUTPUTS_DIR)

    # -- model --
    model_path = MODELS_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)
    logger.info(f"Model saved      -> {model_path}")

    # -- metrics.json --
    metrics = {
        "train":       train_metrics,
        "test":        test_metrics,
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)),
        "n_features":  int(len(feat_cols)),
        "best_iteration": int(best_iter),
    }
    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved    -> {metrics_path}")

    # -- predictions.csv --
    #   Two blocks (train rows + test rows) concatenated with a 'split' label
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    pred_df = pd.concat(
        [
            pd.DataFrame({
                "player":        players_train,
                "actual_elo":    y_train,
                "predicted_elo": train_pred,
                "abs_error":     np.abs(y_train - train_pred),
                "split":         "train",
            }),
            pd.DataFrame({
                "player":        players_test,
                "actual_elo":    y_test,
                "predicted_elo": test_pred,
                "abs_error":     np.abs(y_test - test_pred),
                "split":         "test",
            }),
        ],
        ignore_index=True,
    ).sort_values("abs_error", ascending=False)

    pred_path = OUTPUTS_DIR / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved -> {pred_path}  ({len(pred_df):,} rows)")

    # -- feature_importance.png --
    fi_path = OUTPUTS_DIR / "feature_importance.png"
    _save_importance_plot(feat_cols, model.feature_importances_, fi_path)

    return model, metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Train XGBoost Elo regression model."
    )
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to player_features.parquet "
             "(default: data/processed/player_features.parquet)",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.features) if args.features else None
    model, metrics = train(df)

    print("\n=== Results ===")
    print(f"  Test  RMSE : {metrics['test']['rmse']:.2f} Elo points")
    print(f"  Test  MAE  : {metrics['test']['mae']:.2f} Elo points")
    print(f"  Test  R2   : {metrics['test']['r2']:.4f}")
    print(f"  Best iteration : {metrics['best_iteration']}")
