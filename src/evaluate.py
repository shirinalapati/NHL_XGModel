"""
Model diagnostics and presentation charts for the held-out test season.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.config import OUTPUTS_DIR, ensure_directories
from src.utils import setup_logging

logger = logging.getLogger("nhl_xg.evaluate")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (goals)")
    plt.title("Reliability diagram (test season)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Wrote %s", out_path)


def plot_prob_distribution(y_prob: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(y_prob, bins=40, color="#1f77b4", alpha=0.85, edgecolor="white")
    plt.xlabel("Predicted goal probability (xG)")
    plt.ylabel("Shots")
    plt.title("Distribution of per-shot xG (test season)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_shot_map(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    goals = df["is_goal"] == 1
    plt.scatter(df.loc[~goals, "x_coord"], df.loc[~goals, "y_coord"], s=8, alpha=0.25, label="No goal")
    plt.scatter(df.loc[goals, "x_coord"], df.loc[goals, "y_coord"], s=18, alpha=0.85, label="Goal")
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("NHL x (ft)")
    plt.ylabel("NHL y (ft)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_goals_vs_xg_player(df: pd.DataFrame, out_path: Path, min_shots: int = 30) -> None:
    g = df.groupby("shooter_id", as_index=False).agg(goals=("is_goal", "sum"), xg=("xg", "sum"), shots=("xg", "count"))
    g = g[g["shots"] >= min_shots]
    plt.figure(figsize=(6, 6))
    plt.scatter(g["xg"], g["goals"], alpha=0.65)
    max_val = float(max(g["xg"].max(), g["goals"].max()))
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("xG")
    plt.ylabel("Goals")
    plt.title(f"Player finishing vs shot quality (min {min_shots} shots)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate xG model on held-out test shots.")
    parser.add_argument("--preds", default=str(OUTPUTS_DIR / "test_shot_predictions.parquet"))
    args = parser.parse_args()

    setup_logging()
    ensure_directories()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise SystemExit(f"Missing predictions parquet: {preds_path}. Run `python -m src.train` first.")

    df = pd.read_parquet(preds_path)
    y_true = df["is_goal"].astype(int).to_numpy()
    y_prob = df["xg"].astype(float).to_numpy()

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_true, y_prob)),
        "test_log_loss": float(log_loss(y_true, y_prob)),
        "test_brier": float(brier_score_loss(y_true, y_prob)),
    }
    (OUTPUTS_DIR / "test_metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Test metrics: %s", metrics)

    plot_calibration(y_true, y_prob, OUTPUTS_DIR / "fig_calibration.png")
    plot_prob_distribution(y_prob, OUTPUTS_DIR / "fig_xg_distribution.png")
    plot_shot_map(df, OUTPUTS_DIR / "fig_shot_map_test.png", "Shot map (test season, raw coordinates)")
    plot_goals_vs_xg_player(df, OUTPUTS_DIR / "fig_goals_vs_xg_players.png", min_shots=30)


if __name__ == "__main__":
    main()
