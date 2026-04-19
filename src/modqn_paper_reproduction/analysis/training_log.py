"""Training-log analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._common import plt


def window_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = (
        "scalar_reward",
        "r1_mean",
        "r2_mean",
        "r3_mean",
        "total_handovers",
    )
    return {
        key: float(sum(float(row[key]) for row in rows) / len(rows))
        for key in keys
    }


def summarize_training_log(
    training_log: list[dict[str, Any]],
    *,
    window_size: int = 500,
) -> dict[str, Any]:
    """Summarize first/last-window behavior for a completed training log."""
    if not training_log:
        return {
            "window_size": 0,
            "first_window": {},
            "last_window": {},
            "best_scalar_episode": None,
            "best_scalar_reward": None,
            "final_episode": None,
            "final_scalar_reward": None,
        }

    n = min(window_size, len(training_log))
    first = training_log[:n]
    last = training_log[-n:]
    best = max(training_log, key=lambda row: float(row["scalar_reward"]))
    final = training_log[-1]
    return {
        "window_size": n,
        "first_window": window_means(first),
        "last_window": window_means(last),
        "best_scalar_episode": int(best["episode"]),
        "best_scalar_reward": float(best["scalar_reward"]),
        "final_episode": int(final["episode"]),
        "final_scalar_reward": float(final["scalar_reward"]),
    }


def export_training_log_artifacts(
    output_dir: str | Path,
    *,
    training_log: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write machine-readable training summaries and plots."""
    out_dir = Path(output_dir)
    training_dir = out_dir / "training"
    figures_dir = out_dir / "figures"

    df = pd.DataFrame(training_log)
    episode_metrics_path = training_dir / "episode_metrics.csv"
    loss_curves_path = training_dir / "loss_curves.csv"
    scalar_plot_path = figures_dir / "training-scalar-reward.png"
    objectives_plot_path = figures_dir / "training-objectives.png"

    training_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(episode_metrics_path, index=False)
    df[["episode", "losses"]].assign(
        loss_q1=df["losses"].apply(lambda x: x[0]),
        loss_q2=df["losses"].apply(lambda x: x[1]),
        loss_q3=df["losses"].apply(lambda x: x[2]),
    )[["episode", "loss_q1", "loss_q2", "loss_q3"]].to_csv(loss_curves_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["episode"], df["scalar_reward"], label="scalar_reward")
    ax.set_title("Training Scalar Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Scalar Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(scalar_plot_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["episode"], df["r1_mean"], label="r1_mean")
    ax.plot(df["episode"], df["r2_mean"], label="r2_mean")
    ax.plot(df["episode"], df["r3_mean"], label="r3_mean")
    ax.set_title("Training Objectives")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Objective Mean")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(objectives_plot_path, dpi=150)
    plt.close(fig)

    return {
        "episode_metrics_csv": episode_metrics_path,
        "loss_curves_csv": loss_curves_path,
        "training_scalar_png": scalar_plot_path,
        "training_objectives_png": objectives_plot_path,
    }


__all__ = [
    "export_training_log_artifacts",
    "summarize_training_log",
    "window_means",
]
