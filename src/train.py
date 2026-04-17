"""
Train a scikit-learn logistic regression xG model with a time-based split.

Train: full seasons in `TRAIN_SEASON_IDS_FULL` plus early `TRAIN_PARTIAL_SEASON_ID`
      through `TRAIN_PARTIAL_CUTOFF_DATE`.
Test:  `TEST_SEASON_ID` games on/after `TEST_START_DATE`.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, ensure_directories
from src.utils import setup_logging

logger = logging.getLogger("nhl_xg.train")

NUMERIC_FEATURES = [
    "shot_distance",
    "shot_angle",
    "abs_shot_angle",
    "time_since_prev_event_seconds",
    "rebound_flag",
    "rush_flag",
    "same_team_prev_shot_flag",
    "is_pp",
    "is_sh",
    "is_ev",
    "is_3v3",
    "empty_net",
    "home_away_home",
    "period",
    "score_diff_shooter",
]

CATEGORICAL_FEATURES = [
    "shot_type",
]


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    clf = LogisticRegression(
        max_iter=500,
        C=1.0,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def safe_odds_ratio(c: float) -> float:
    return float(np.exp(c))


def extract_coefficients(pipe: Pipeline) -> pd.DataFrame:
    clf: LogisticRegression = pipe.named_steps["clf"]
    pre: ColumnTransformer = pipe.named_steps["prep"]
    num_names = NUMERIC_FEATURES
    cat_encoder: OneHotEncoder = pre.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    names = num_names + cat_names
    coefs = clf.coef_.ravel()
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["odds_ratio"] = df["coefficient"].map(safe_odds_ratio)
    return df.iloc[np.argsort(-np.abs(df["coefficient"].to_numpy()))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic regression xG model.")
    parser.add_argument("--data", default=str(OUTPUTS_DIR / "modeling_dataset.parquet"))
    args = parser.parse_args()

    setup_logging()
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    train_df = df[df["dataset_split"] == "train"].copy()
    test_df = df[df["dataset_split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        vc = df["dataset_split"].value_counts(dropna=False).to_dict()
        raise SystemExit(
            "Train or test split is empty.\n"
            f"dataset_split counts: {vc}\n"
            "Most common cause: ingestion did not cover the test window dates in `src/config.py` "
            "(for 2025–26, you need games on/after the configured test start date)."
        )

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["is_goal"].astype(int)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["is_goal"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    model_path = MODELS_DIR / "xg_logistic.joblib"
    joblib.dump(pipe, model_path)
    logger.info("Saved model: %s", model_path)

    coef_df = extract_coefficients(pipe)
    coef_path = OUTPUTS_DIR / "coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    logger.info("Saved coefficients: %s", coef_path)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_goals": int(y_train.sum()),
        "test_goals": int(y_test.sum()),
        "train_roc_auc": float(roc_auc_score(y_train, p_train)),
        "test_roc_auc": float(roc_auc_score(y_test, p_test)),
        "train_log_loss": float(log_loss(y_train, p_train)),
        "test_log_loss": float(log_loss(y_test, p_test)),
        "train_brier": float(brier_score_loss(y_train, p_train)),
        "test_brier": float(brier_score_loss(y_test, p_test)),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics: %s", metrics)

    base_cols = [
        "game_id",
        "event_id",
        "season",
        "game_date",
        "team_id",
        "shooter_id",
        "is_goal",
        "x_coord",
        "y_coord",
        "shot_distance",
        "shot_angle",
    ]
    if "team_abbr" in test_df.columns:
        base_cols.append("team_abbr")
    out_preds = test_df[base_cols].copy()
    out_preds["xg"] = p_test
    pred_path = OUTPUTS_DIR / "test_shot_predictions.parquet"
    out_preds.to_parquet(pred_path, index=False)
    logger.info("Saved test predictions: %s", pred_path)


if __name__ == "__main__":
    main()
