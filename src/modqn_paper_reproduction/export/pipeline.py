"""Sweep/export helpers for Phase 01 artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

_MPL_CACHE_DIR = Path("/tmp/modqn-mpl-cache")
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .replay_bundle import export_replay_bundle, validate_replay_bundle


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _weight_label(row: tuple[float, float, float]) -> str:
    return "/".join(f"{x:.1f}" for x in row)


def _safe_abs_scale(value: float, floor: float = 1e-12) -> float:
    """Avoid zero or near-zero normalization scales."""
    return max(abs(float(value)), floor)


def _window_means(rows: list[dict[str, Any]]) -> dict[str, float]:
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


def _summarize_training_log(
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
        "first_window": _window_means(first),
        "last_window": _window_means(last),
        "best_scalar_episode": int(best["episode"]),
        "best_scalar_reward": float(best["scalar_reward"]),
        "final_episode": int(final["episode"]),
        "final_scalar_reward": float(final["scalar_reward"]),
    }


def _build_table_ii_analysis_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build winner, spread, and delta-vs-MODQN frames from Table II results."""
    winners: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    tie_tolerance = 1e-9

    for weight_label, grp in df.groupby("weight_label", sort=False):
        ordered = grp.sort_values(
            ["mean_scalar_reward", "method"],
            ascending=[False, True],
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        tied_best = ordered[
            np.isclose(
                ordered["mean_scalar_reward"],
                float(best["mean_scalar_reward"]),
                atol=tie_tolerance,
                rtol=0.0,
            )
        ]
        remaining = ordered[
            ~np.isclose(
                ordered["mean_scalar_reward"],
                float(best["mean_scalar_reward"]),
                atol=tie_tolerance,
                rtol=0.0,
            )
        ]
        runner_up = remaining.iloc[0] if not remaining.empty else tied_best.iloc[0]
        winners.append(
            {
                "weight_label": weight_label,
                "best_method": best["method"],
                "best_methods": "|".join(str(m) for m in tied_best["method"]),
                "num_tied_best": int(len(tied_best)),
                "is_exact_tie": bool(len(tied_best) > 1),
                "best_mean_scalar": float(best["mean_scalar_reward"]),
                "runner_up_method": runner_up["method"],
                "runner_up_mean_scalar": float(runner_up["mean_scalar_reward"]),
                "margin_to_runner_up": float(
                    best["mean_scalar_reward"] - runner_up["mean_scalar_reward"]
                ),
            }
        )
        spreads.append(
            {
                "weight_label": weight_label,
                "scalar_spread": float(grp["mean_scalar_reward"].max() - grp["mean_scalar_reward"].min()),
                "r1_spread": float(grp["mean_r1"].max() - grp["mean_r1"].min()),
                "r2_spread": float(grp["mean_r2"].max() - grp["mean_r2"].min()),
                "r3_spread": float(grp["mean_r3"].max() - grp["mean_r3"].min()),
                "handover_spread": float(
                    grp["mean_total_handovers"].max() - grp["mean_total_handovers"].min()
                ),
            }
        )

        modqn_row = grp.loc[grp["method"] == "MODQN"]
        if modqn_row.empty:
            continue
        modqn = modqn_row.iloc[0]
        for _, row in grp.iterrows():
            deltas.append(
                {
                    "weight_label": weight_label,
                    "method": row["method"],
                    "mean_scalar_reward": float(row["mean_scalar_reward"]),
                    "scalar_delta_vs_modqn": float(row["mean_scalar_reward"] - modqn["mean_scalar_reward"]),
                    "r1_delta_vs_modqn": float(row["mean_r1"] - modqn["mean_r1"]),
                    "r2_delta_vs_modqn": float(row["mean_r2"] - modqn["mean_r2"]),
                    "r3_delta_vs_modqn": float(row["mean_r3"] - modqn["mean_r3"]),
                    "handover_delta_vs_modqn": float(
                        row["mean_total_handovers"] - modqn["mean_total_handovers"]
                    ),
                    "is_row_winner": bool(row["method"] == best["method"]),
                }
            )

    return pd.DataFrame(winners), pd.DataFrame(spreads), pd.DataFrame(deltas)


def _write_table_ii_analysis_markdown(
    path: Path,
    *,
    df: pd.DataFrame,
    winners_df: pd.DataFrame,
    spreads_df: pd.DataFrame,
    reference_summary: dict[str, Any] | None,
) -> Path:
    """Write a concise markdown analysis note for Table II outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tie_inclusive_counts: dict[str, int] = {}
    exact_tie_rows = 0
    for _, row in winners_df.iterrows():
        methods = str(row["best_methods"]).split("|")
        if bool(row["is_exact_tie"]):
            exact_tie_rows += 1
        for method in methods:
            tie_inclusive_counts[method] = tie_inclusive_counts.get(method, 0) + 1
    max_scalar_spread = float(spreads_df["scalar_spread"].max()) if not spreads_df.empty else 0.0
    max_r1_spread = float(spreads_df["r1_spread"].max()) if not spreads_df.empty else 0.0
    max_r2_spread = float(spreads_df["r2_spread"].max()) if not spreads_df.empty else 0.0
    max_r3_spread = float(spreads_df["r3_spread"].max()) if not spreads_df.empty else 0.0
    max_handover_spread = float(spreads_df["handover_spread"].max()) if not spreads_df.empty else 0.0
    training_episodes = sorted({int(x) for x in df["training_episodes"].unique() if int(x) > 0})

    lines = [
        "# Table II Analysis",
        "",
        "## Summary",
        "",
        f"- Weight rows: `{df['weight_label'].nunique()}`",
        f"- Methods: `{', '.join(sorted(df['method'].unique()))}`",
        f"- Learned-method training episodes observed: `{training_episodes}`",
        f"- Tie-inclusive best/tied-best counts: `{tie_inclusive_counts}`",
        f"- Exact tie rows: `{exact_tie_rows}`",
        "",
        "## Objective Spread",
        "",
        f"- max scalar spread across methods: `{max_scalar_spread:.6f}`",
        f"- max r1 spread across methods: `{max_r1_spread:.6f}`",
        f"- max r2 spread across methods: `{max_r2_spread:.6f}`",
        f"- max r3 spread across methods: `{max_r3_spread:.6f}`",
        f"- max handover spread across methods: `{max_handover_spread:.6f}`",
        "",
        "## Interpretation",
        "",
        "- This Table II surface is near-tied across methods.",
        "- The dominant cross-method variation is in `r2` / handover count, not `r1` or `r3`.",
    ]

    if reference_summary is not None:
        first = reference_summary.get("first_window", {})
        last = reference_summary.get("last_window", {})
        lines.extend(
            [
                "",
                "## Long-Run Reference",
                "",
                f"- first-window scalar: `{first.get('scalar_reward')}`",
                f"- last-window scalar: `{last.get('scalar_reward')}`",
                f"- first-window r1/r2/r3: `{first.get('r1_mean')}`, `{first.get('r2_mean')}`, `{first.get('r3_mean')}`",
                f"- last-window r1/r2/r3: `{last.get('r1_mean')}`, `{last.get('r2_mean')}`, `{last.get('r3_mean')}`",
                "",
                "- The long-run reference shows throughput collapse and load-balance deterioration while handover penalty improves.",
                "- Together with the near-tied Table II rows, this supports the current reward-dominance interpretation.",
            ]
        )

    path.write_text("\n".join(lines) + "\n")
    return path


def _collect_reward_diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Collect the repo-local reward-scale diagnostics from the environment surface."""
    from ..config_loader import build_environment, get_seeds

    seeds = get_seeds(cfg)
    env = build_environment(cfg)
    env_rng = np.random.default_rng(int(seeds["environment_seed"]))
    mobility_rng = np.random.default_rng(int(seeds["mobility_seed"]))
    _states, _masks, diag = env.reset(env_rng, mobility_rng)
    payload = asdict(diag)
    payload["train_seed"] = int(seeds["train_seed"])
    payload["environment_seed"] = int(seeds["environment_seed"])
    payload["mobility_seed"] = int(seeds["mobility_seed"])
    return payload


def _build_reward_geometry_scale_table(
    diagnostics: dict[str, Any],
    reference_summary: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build named normalization scenarios for reward-geometry analysis."""
    rows = [
        {
            "scenario": "raw-unscaled",
            "source": "identity",
            "scale_r1": 1.0,
            "scale_r2": 1.0,
            "scale_r3": 1.0,
        },
        {
            "scenario": "diagnostic-abs",
            "source": "env-diagnostics",
            "scale_r1": _safe_abs_scale(diagnostics["sample_r1"]),
            "scale_r2": _safe_abs_scale(diagnostics["sample_r2_beam_change"]),
            "scale_r3": _safe_abs_scale(diagnostics["sample_r3"]),
        },
    ]

    if reference_summary is not None:
        first = reference_summary.get("first_window", {})
        last = reference_summary.get("last_window", {})
        if first:
            rows.append(
                {
                    "scenario": "reference-first-window-abs",
                    "source": "reference-run:first-window",
                    "scale_r1": _safe_abs_scale(first["r1_mean"]),
                    "scale_r2": _safe_abs_scale(first["r2_mean"]),
                    "scale_r3": _safe_abs_scale(first["r3_mean"]),
                }
            )
        if last:
            rows.append(
                {
                    "scenario": "reference-last-window-abs",
                    "source": "reference-run:last-window",
                    "scale_r1": _safe_abs_scale(last["r1_mean"]),
                    "scale_r2": _safe_abs_scale(last["r2_mean"]),
                    "scale_r3": _safe_abs_scale(last["r3_mean"]),
                }
            )
    return pd.DataFrame(rows)


def _build_reward_geometry_table_ii_frames(
    table_df: pd.DataFrame,
    scale_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-score Table II rows under experimental normalization scenarios."""
    scenario_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    tie_tolerance = 1e-9

    for _, scale_row in scale_df.iterrows():
        scenario = str(scale_row["scenario"])
        scale_r1 = float(scale_row["scale_r1"])
        scale_r2 = float(scale_row["scale_r2"])
        scale_r3 = float(scale_row["scale_r3"])

        scenario_df = table_df.copy()
        scenario_df["scenario"] = scenario
        scenario_df["normalized_r1"] = scenario_df["mean_r1"] / scale_r1
        scenario_df["normalized_r2"] = scenario_df["mean_r2"] / scale_r2
        scenario_df["normalized_r3"] = scenario_df["mean_r3"] / scale_r3
        scenario_df["scenario_scalar"] = (
            scenario_df["w1"] * scenario_df["normalized_r1"]
            + scenario_df["w2"] * scenario_df["normalized_r2"]
            + scenario_df["w3"] * scenario_df["normalized_r3"]
        )
        scenario_df["normalized_abs_contribution_r1"] = (
            scenario_df["w1"] * scenario_df["normalized_r1"].abs()
        )
        scenario_df["normalized_abs_contribution_r2"] = (
            scenario_df["w2"] * scenario_df["normalized_r2"].abs()
        )
        scenario_df["normalized_abs_contribution_r3"] = (
            scenario_df["w3"] * scenario_df["normalized_r3"].abs()
        )
        total_abs = (
            scenario_df["normalized_abs_contribution_r1"]
            + scenario_df["normalized_abs_contribution_r2"]
            + scenario_df["normalized_abs_contribution_r3"]
        ).replace(0.0, 1.0)
        scenario_df["normalized_share_r1"] = (
            scenario_df["normalized_abs_contribution_r1"] / total_abs
        )
        scenario_df["normalized_share_r2"] = (
            scenario_df["normalized_abs_contribution_r2"] / total_abs
        )
        scenario_df["normalized_share_r3"] = (
            scenario_df["normalized_abs_contribution_r3"] / total_abs
        )
        scenario_rows.extend(scenario_df.to_dict(orient="records"))

        for weight_label, grp in scenario_df.groupby("weight_label", sort=False):
            ordered = grp.sort_values(
                ["scenario_scalar", "method"],
                ascending=[False, True],
            ).reset_index(drop=True)
            best = ordered.iloc[0]
            tied_best = ordered[
                np.isclose(
                    ordered["scenario_scalar"],
                    float(best["scenario_scalar"]),
                    atol=tie_tolerance,
                    rtol=0.0,
                )
            ]
            remaining = ordered[
                ~np.isclose(
                    ordered["scenario_scalar"],
                    float(best["scenario_scalar"]),
                    atol=tie_tolerance,
                    rtol=0.0,
                )
            ]
            runner_up = remaining.iloc[0] if not remaining.empty else tied_best.iloc[0]
            winner_rows.append(
                {
                    "scenario": scenario,
                    "weight_label": weight_label,
                    "best_method": best["method"],
                    "best_methods": "|".join(str(m) for m in tied_best["method"]),
                    "num_tied_best": int(len(tied_best)),
                    "best_scenario_scalar": float(best["scenario_scalar"]),
                    "runner_up_method": runner_up["method"],
                    "runner_up_scenario_scalar": float(runner_up["scenario_scalar"]),
                    "margin_to_runner_up": float(
                        best["scenario_scalar"] - runner_up["scenario_scalar"]
                    ),
                }
            )

    return pd.DataFrame(scenario_rows), pd.DataFrame(winner_rows)


def _build_reference_window_contributions(
    reference_summary: dict[str, Any] | None,
    scale_df: pd.DataFrame,
    *,
    baseline_weights: tuple[float, float, float],
) -> pd.DataFrame:
    """Build raw and normalized contribution tables for reference-run windows."""
    if reference_summary is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    windows = {
        "first_window": reference_summary.get("first_window", {}),
        "last_window": reference_summary.get("last_window", {}),
    }
    for window_name, window in windows.items():
        if not window:
            continue
        r1 = float(window["r1_mean"])
        r2 = float(window["r2_mean"])
        r3 = float(window["r3_mean"])
        raw_c1 = baseline_weights[0] * r1
        raw_c2 = baseline_weights[1] * r2
        raw_c3 = baseline_weights[2] * r3
        rows.append(
            {
                "window": window_name,
                "scenario": "raw-unscaled",
                "scalar": raw_c1 + raw_c2 + raw_c3,
                "contribution_r1": raw_c1,
                "contribution_r2": raw_c2,
                "contribution_r3": raw_c3,
            }
        )
        for _, scale_row in scale_df.iterrows():
            scenario = str(scale_row["scenario"])
            if scenario == "raw-unscaled":
                continue
            c1 = baseline_weights[0] * (r1 / float(scale_row["scale_r1"]))
            c2 = baseline_weights[1] * (r2 / float(scale_row["scale_r2"]))
            c3 = baseline_weights[2] * (r3 / float(scale_row["scale_r3"]))
            rows.append(
                {
                    "window": window_name,
                    "scenario": scenario,
                    "scalar": c1 + c2 + c3,
                    "contribution_r1": c1,
                    "contribution_r2": c2,
                    "contribution_r3": c3,
                }
            )
    return pd.DataFrame(rows)


def export_reward_geometry_analysis(
    output_dir: str | Path,
    *,
    cfg: dict[str, Any],
    table_ii_dir: str | Path,
    reference_run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Export a reward-geometry analysis bundle from existing artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = out_dir / "analysis"
    evaluation_dir = out_dir / "evaluation"
    figures_dir = out_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_dir = Path(table_ii_dir)
    table_json_path = table_dir / "evaluation" / "table-ii.json"
    table_payload = json.loads(table_json_path.read_text())
    table_df = pd.DataFrame(table_payload["rows"])

    reference_summary = None
    if reference_run_dir is not None:
        ref_dir = Path(reference_run_dir)
        training_log_path = ref_dir / "training_log.json"
        if training_log_path.exists():
            reference_summary = _summarize_training_log(
                json.loads(training_log_path.read_text())
            )
            reference_summary["reference_run_dir"] = str(ref_dir)

    diagnostics = _collect_reward_diagnostics(cfg)
    scale_df = _build_reward_geometry_scale_table(diagnostics, reference_summary)
    scenario_df, winner_df = _build_reward_geometry_table_ii_frames(table_df, scale_df)

    baseline_weights = tuple(
        float(x) for x in cfg.get("baseline", cfg).get("objective_weights", [0.5, 0.3, 0.2])
    )
    reference_window_df = _build_reference_window_contributions(
        reference_summary,
        scale_df,
        baseline_weights=baseline_weights,
    )

    diagnostics_path = _write_json(evaluation_dir / "reward-geometry-diagnostics.json", diagnostics)
    scales_csv_path = evaluation_dir / "reward-geometry-scales.csv"
    scale_df.to_csv(scales_csv_path, index=False)
    scenario_csv_path = evaluation_dir / "reward-geometry-table-ii.csv"
    scenario_df.to_csv(scenario_csv_path, index=False)
    winner_csv_path = evaluation_dir / "reward-geometry-winners.csv"
    winner_df.to_csv(winner_csv_path, index=False)
    reference_window_csv_path = evaluation_dir / "reward-geometry-reference-window-contributions.csv"
    if reference_window_df.empty:
        pd.DataFrame(
            columns=["window", "scenario", "scalar", "contribution_r1", "contribution_r2", "contribution_r3"]
        ).to_csv(reference_window_csv_path, index=False)
    else:
        reference_window_df.to_csv(reference_window_csv_path, index=False)

    summary_path = _write_json(
        evaluation_dir / "reward-geometry-summary.json",
        {
            "diagnostics": diagnostics,
            "scale_scenarios": scale_df.to_dict(orient="records"),
            "baseline_weights": list(baseline_weights),
            "table_ii_input_dir": str(table_dir),
            "reference_summary": reference_summary,
        },
    )

    spread_plot_path = figures_dir / "reward-geometry-scalar-spread.png"
    spread_df = (
        scenario_df.groupby(["scenario", "weight_label"], as_index=False)["scenario_scalar"]
        .agg(lambda col: float(col.max() - col.min()))
        .rename(columns={"scenario_scalar": "scalar_spread"})
    )
    pivot = spread_df.pivot(index="weight_label", columns="scenario", values="scalar_spread")
    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.set_title("Reward-Geometry Scalar Spread by Scenario")
    ax.set_xlabel("Weight Row")
    ax.set_ylabel("Scalar Spread Across Methods")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(spread_plot_path, dpi=150)
    plt.close()

    analysis_md_path = analysis_dir / "reward-geometry.md"
    max_scenario_spread = (
        spread_df.sort_values("scalar_spread", ascending=False).iloc[0].to_dict()
        if not spread_df.empty
        else {}
    )
    tie_rows = int((winner_df["num_tied_best"] > 1).sum()) if not winner_df.empty else 0
    lines = [
        "# Reward Geometry Analysis",
        "",
        "## Diagnostics",
        "",
        f"- sample r1: `{diagnostics['sample_r1']}`",
        f"- sample r2 (beam change): `{diagnostics['sample_r2_beam_change']}`",
        f"- sample r3: `{diagnostics['sample_r3']}`",
        f"- r1/r2 ratio: `{diagnostics['r1_r2_ratio']}`",
        f"- r1/r3 ratio: `{diagnostics['r1_r3_ratio']}`",
        "",
        "## Scenario Summary",
        "",
        f"- scenarios: `{list(scale_df['scenario'])}`",
        f"- exact tie rows across all scenarios: `{tie_rows}`",
        f"- max scalar spread record: `{max_scenario_spread}`",
        "",
        "## Interpretation",
        "",
        "- This surface is explicitly experimental and does not change the baseline training rule.",
        "- It re-scores existing Table II outputs under alternative normalization scales.",
        "- If winners remain near-tied after normalization, the bottleneck is not just scalar weighting but the policy/evaluation regime itself.",
    ]
    if reference_summary is not None:
        lines.extend(
            [
                "",
                "## Long-Run Linkage",
                "",
                f"- first-window summary: `{reference_summary.get('first_window')}`",
                f"- last-window summary: `{reference_summary.get('last_window')}`",
                "- The long-run reference is included to connect calibration questions to the existing reward-dominance and objective-drift findings.",
            ]
        )
    analysis_md_path.write_text("\n".join(lines) + "\n")

    return {
        "reward_geometry_diagnostics_json": diagnostics_path,
        "reward_geometry_scales_csv": scales_csv_path,
        "reward_geometry_table_ii_csv": scenario_csv_path,
        "reward_geometry_winners_csv": winner_csv_path,
        "reward_geometry_reference_window_csv": reference_window_csv_path,
        "reward_geometry_summary_json": summary_path,
        "reward_geometry_scalar_spread_png": spread_plot_path,
        "reward_geometry_md": analysis_md_path,
    }


def export_training_run(input_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Export a completed run artifact into CSV/PNG bundle surfaces."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    metadata = json.loads((in_dir / "run_metadata.json").read_text())
    training_log = json.loads((in_dir / "training_log.json").read_text())

    training_dir = out_dir / "training"
    evaluation_dir = out_dir / "evaluation"
    sweeps_dir = evaluation_dir / "sweeps"
    figures_dir = out_dir / "figures"

    df = pd.DataFrame(training_log)
    episode_metrics_path = training_dir / "episode_metrics.csv"
    loss_curves_path = training_dir / "loss_curves.csv"
    training_dir.mkdir(parents=True, exist_ok=True)
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(episode_metrics_path, index=False)
    df[["episode", "losses"]].assign(
        loss_q1=df["losses"].apply(lambda x: x[0]),
        loss_q2=df["losses"].apply(lambda x: x[1]),
        loss_q3=df["losses"].apply(lambda x: x[2]),
    )[["episode", "loss_q1", "loss_q2", "loss_q3"]].to_csv(loss_curves_path, index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    scalar_plot_path = figures_dir / "training-scalar-reward.png"
    objectives_plot_path = figures_dir / "training-objectives.png"

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

    assumptions_path = _write_json(
        out_dir / "assumptions.json",
        metadata.get("resolved_assumptions", {}),
    )
    replay_outputs = export_replay_bundle(
        in_dir,
        out_dir,
        metadata=metadata,
    )
    summary_path = _write_json(
        evaluation_dir / "summary.json",
        {
            "paper_id": metadata.get("paper_id"),
            "config_path": metadata.get("config_path"),
            "checkpoint_rule": metadata.get("checkpoint_rule"),
            "checkpoint_files": metadata.get("checkpoint_files"),
            "best_eval_summary": metadata.get("best_eval_summary"),
            "training_summary": metadata.get("training_summary"),
            "bundle_schema_version": replay_outputs["bundle_schema_version"],
            "replay_timeline": replay_outputs["replay_summary"],
        },
    )
    validate_replay_bundle(out_dir)

    return {
        "manifest": replay_outputs["manifest"],
        "assumptions": assumptions_path,
        "config_resolved_json": replay_outputs["config_resolved"],
        "provenance_map_json": replay_outputs["provenance_map"],
        "timeline_step_trace_jsonl": replay_outputs["timeline_step_trace"],
        "episode_metrics_csv": episode_metrics_path,
        "loss_curves_csv": loss_curves_path,
        "summary_json": summary_path,
        "training_scalar_png": scalar_plot_path,
        "training_objectives_png": objectives_plot_path,
    }


def export_figure_sweep_results(
    output_dir: str | Path,
    *,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    suite_spec: dict[str, Any],
    reference_run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write a Fig. 3 to Fig. 6 sweep bundle with detail and weighted outputs."""
    out_dir = Path(output_dir)
    figures_dir = out_dir / "figures"
    evaluation_dir = out_dir / "evaluation"
    analysis_dir = out_dir / "analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(["parameter_value", "method"]).reset_index(drop=True)
    suite = str(suite_spec["suite"])
    figure_id = str(suite_spec["figure_id"])
    parameter_label = str(suite_spec["parameter_label"])
    parameter_unit = str(suite_spec["parameter_unit"])
    point_count = int(df["parameter_value"].nunique()) if not df.empty else 0
    analysis_context = {
        "pointSelectionMode": manifest.get("pointSelectionMode"),
        "configuredSweepPointSet": manifest.get("configuredSweepPointSet"),
        "requestedSweepPointSet": manifest.get("requestedSweepPointSet"),
        "effectiveSweepPointSet": manifest.get("sweepPointSet"),
    }

    full_json_path = _write_json(
        evaluation_dir / f"{suite}.json",
        {
            "rows": rows,
            "analysisContext": analysis_context,
        },
    )
    detail_csv_path = evaluation_dir / f"{suite}-detail.csv"
    weighted_csv_path = evaluation_dir / f"{suite}-weighted-reward.csv"
    winners_csv_path = evaluation_dir / f"{suite}-weighted-winners.csv"
    manifest_path = _write_json(out_dir / "manifest.json", manifest)
    analysis_md_path = analysis_dir / f"{suite}-analysis.md"
    detail_plot_path = figures_dir / f"{suite}-objectives.png"
    weighted_plot_path = figures_dir / f"{suite}-weighted-reward.png"

    detail_columns = [
        "suite",
        "figure_id",
        "parameter_name",
        "parameter_label",
        "parameter_unit",
        "parameter_value",
        "method",
        "mean_r1",
        "std_r1",
        "mean_r2",
        "std_r2",
        "mean_r3",
        "std_r3",
        "mean_total_handovers",
        "std_total_handovers",
        "policy_episode",
        "eval_seed_count",
        "training_episodes",
        "checkpoint_kind",
    ]
    weighted_columns = [
        "suite",
        "figure_id",
        "parameter_name",
        "parameter_label",
        "parameter_unit",
        "parameter_value",
        "method",
        "weight_label",
        "w1",
        "w2",
        "w3",
        "mean_scalar_reward",
        "std_scalar_reward",
        "policy_episode",
        "eval_seed_count",
        "training_episodes",
        "checkpoint_kind",
    ]
    df[detail_columns].to_csv(detail_csv_path, index=False)
    df[weighted_columns].to_csv(weighted_csv_path, index=False)

    winner_rows: list[dict[str, Any]] = []
    tie_inclusive_counts: dict[str, int] = {}
    tie_tolerance = 1e-9
    for parameter_value, grp in df.groupby("parameter_value", sort=True):
        ordered = grp.sort_values(
            ["mean_scalar_reward", "method"],
            ascending=[False, True],
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        tied_best = ordered[
            np.isclose(
                ordered["mean_scalar_reward"],
                float(best["mean_scalar_reward"]),
                atol=tie_tolerance,
                rtol=0.0,
            )
        ]
        remaining = ordered[
            ~np.isclose(
                ordered["mean_scalar_reward"],
                float(best["mean_scalar_reward"]),
                atol=tie_tolerance,
                rtol=0.0,
            )
        ]
        runner_up = remaining.iloc[0] if not remaining.empty else tied_best.iloc[0]
        for method in tied_best["method"]:
            method_str = str(method)
            tie_inclusive_counts[method_str] = tie_inclusive_counts.get(method_str, 0) + 1
        winner_rows.append(
            {
                "suite": suite,
                "figure_id": figure_id,
                "parameter_label": parameter_label,
                "parameter_unit": parameter_unit,
                "parameter_value": float(parameter_value),
                "best_method": best["method"],
                "best_methods": "|".join(str(m) for m in tied_best["method"]),
                "num_tied_best": int(len(tied_best)),
                "best_mean_scalar": float(best["mean_scalar_reward"]),
                "runner_up_method": runner_up["method"],
                "runner_up_mean_scalar": float(runner_up["mean_scalar_reward"]),
                "margin_to_runner_up": float(
                    best["mean_scalar_reward"] - runner_up["mean_scalar_reward"]
                ),
            }
        )
    winners_df = pd.DataFrame(winner_rows)
    winners_df.to_csv(winners_csv_path, index=False)

    if not df.empty:
        methods = list(df["method"].drop_duplicates())
        detail_df = df.sort_values(["parameter_value", "method"]).reset_index(drop=True)

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        metric_specs = [
            ("mean_r1", "Throughput (r1)"),
            ("mean_r2", "Handover Cost (r2)"),
            ("mean_r3", "Load Balance (r3)"),
        ]
        for axis, (metric, title) in zip(axes, metric_specs):
            for method in methods:
                grp = detail_df.loc[detail_df["method"] == method].sort_values("parameter_value")
                axis.plot(
                    grp["parameter_value"],
                    grp[metric],
                    marker="o",
                    label=method,
                )
            axis.set_title(title)
            axis.set_ylabel("Mean Reward")
            axis.grid(True, alpha=0.3)
        axes[-1].set_xlabel(f"{parameter_label} ({parameter_unit})" if parameter_unit else parameter_label)
        axes[0].legend()
        fig.suptitle(f"{figure_id}: Detailed Performance of Three Objectives", y=0.995)
        fig.tight_layout()
        fig.savefig(detail_plot_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for method in methods:
            grp = detail_df.loc[detail_df["method"] == method].sort_values("parameter_value")
            ax.plot(
                grp["parameter_value"],
                grp["mean_scalar_reward"],
                marker="o",
                label=method,
            )
        ax.set_title(f"{figure_id}: Weighted Reward")
        ax.set_xlabel(f"{parameter_label} ({parameter_unit})" if parameter_unit else parameter_label)
        ax.set_ylabel("Mean Weighted Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(weighted_plot_path, dpi=150)
        plt.close(fig)

    reference_summary_path = None
    reference_summary = None
    if reference_run_dir is not None:
        ref_dir = Path(reference_run_dir)
        training_log_path = ref_dir / "training_log.json"
        if training_log_path.exists():
            reference_summary = _summarize_training_log(
                json.loads(training_log_path.read_text())
            )
            reference_summary["reference_run_dir"] = str(ref_dir)
            reference_summary_path = evaluation_dir / f"{suite}-long-run-reference-summary.json"
            _write_json(reference_summary_path, reference_summary)

    modqn_df = df.loc[df["method"] == "MODQN"].sort_values("parameter_value")
    modqn_first_scalar = (
        float(modqn_df.iloc[0]["mean_scalar_reward"]) if not modqn_df.empty else None
    )
    modqn_last_scalar = (
        float(modqn_df.iloc[-1]["mean_scalar_reward"]) if not modqn_df.empty else None
    )
    exact_tie_rows = int((winners_df["num_tied_best"] > 1).sum()) if not winners_df.empty else 0
    max_scalar_spread = 0.0
    if point_count > 0:
        spread_df = (
            df.groupby("parameter_value", as_index=False)["mean_scalar_reward"]
            .agg(lambda col: float(col.max() - col.min()))
            .rename(columns={"mean_scalar_reward": "scalar_spread"})
        )
        if not spread_df.empty:
            max_scalar_spread = float(spread_df["scalar_spread"].max())

    lines = [
        f"# {figure_id} Analysis",
        "",
        "## Summary",
        "",
        f"- suite: `{suite}`",
        f"- sweep parameter: `{parameter_label}`",
        f"- point count: `{point_count}`",
        f"- point selection mode: `{analysis_context.get('pointSelectionMode')}`",
        f"- configured point set: `{analysis_context.get('configuredSweepPointSet')}`",
        f"- requested point override: `{analysis_context.get('requestedSweepPointSet')}`",
        f"- effective point set: `{analysis_context.get('effectiveSweepPointSet')}`",
        f"- methods: `{', '.join(df['method'].drop_duplicates())}`",
        f"- baseline weight row: `{manifest.get('baselineWeightRow')}`",
        f"- tie-inclusive weighted-reward wins: `{tie_inclusive_counts}`",
        f"- exact tie points: `{exact_tie_rows}`",
        f"- max weighted-reward spread across methods: `{max_scalar_spread:.6f}`",
        "",
        "## MODQN Weighted-Reward Trend",
        "",
        f"- first-point mean scalar: `{modqn_first_scalar}`",
        f"- last-point mean scalar: `{modqn_last_scalar}`",
        "",
        "## Interpretation",
        "",
        "- This figure family reports raw objective means plus weighted reward under the baseline weight row.",
        "- The output is intended as a first executable paper-style sweep surface, not a claim of exact visual parity with the paper figures.",
        "- Use the machine-readable CSV/JSON companions as the authoritative comparison surface.",
    ]
    if reference_summary is not None:
        lines.extend(
            [
                "",
                "## Long-Run Reference",
                "",
                f"- first-window scalar: `{reference_summary.get('first_window', {}).get('scalar_reward')}`",
                f"- last-window scalar: `{reference_summary.get('last_window', {}).get('scalar_reward')}`",
                "- The reference run is included only to connect figure-sweep trends to the known long-run anomaly.",
            ]
        )
    analysis_md_path.write_text("\n".join(lines) + "\n")

    return {
        f"{suite}_json": full_json_path,
        f"{suite}_detail_csv": detail_csv_path,
        f"{suite}_weighted_reward_csv": weighted_csv_path,
        f"{suite}_weighted_winners_csv": winners_csv_path,
        f"{suite}_objectives_png": detail_plot_path,
        f"{suite}_weighted_reward_png": weighted_plot_path,
        f"{suite}_analysis_md": analysis_md_path,
        **(
            {f"{suite}_long_run_reference_summary_json": reference_summary_path}
            if reference_summary_path is not None
            else {}
        ),
        "manifest": manifest_path,
    }


def export_table_ii_results(
    output_dir: str | Path,
    *,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    reference_run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write Table II machine-readable outputs and a scalar-reward plot."""
    out_dir = Path(output_dir)
    figures_dir = out_dir / "figures"
    evaluation_dir = out_dir / "evaluation"
    analysis_dir = out_dir / "analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = figures_dir / "table-ii.csv"
    json_path = evaluation_dir / "table-ii.json"
    plot_path = figures_dir / "table-ii.png"
    manifest_path = out_dir / "manifest.json"
    winners_csv_path = evaluation_dir / "table-ii-winners.csv"
    spreads_csv_path = evaluation_dir / "table-ii-spreads.csv"
    deltas_csv_path = evaluation_dir / "table-ii-deltas-vs-modqn.csv"
    analysis_md_path = analysis_dir / "table-ii-analysis.md"

    df.to_csv(csv_path, index=False)
    _write_json(json_path, {"rows": rows})
    _write_json(manifest_path, manifest)

    winners_df, spreads_df, deltas_df = _build_table_ii_analysis_frames(df)
    winners_df.to_csv(winners_csv_path, index=False)
    spreads_df.to_csv(spreads_csv_path, index=False)
    deltas_df.to_csv(deltas_csv_path, index=False)

    reference_summary_path = None
    reference_summary = None
    if reference_run_dir is not None:
        ref_dir = Path(reference_run_dir)
        training_log_path = ref_dir / "training_log.json"
        if training_log_path.exists():
            reference_training_log = json.loads(training_log_path.read_text())
            reference_summary = _summarize_training_log(reference_training_log)
            reference_summary["reference_run_dir"] = str(ref_dir)
            reference_summary_path = evaluation_dir / "long-run-reference-summary.json"
            _write_json(reference_summary_path, reference_summary)

    _write_table_ii_analysis_markdown(
        analysis_md_path,
        df=df,
        winners_df=winners_df,
        spreads_df=spreads_df,
        reference_summary=reference_summary,
    )

    pivot = df.pivot(index="weight_label", columns="method", values="mean_scalar_reward")
    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.set_title("Table II Mean Scalar Reward by Weight Row")
    ax.set_xlabel("Weight Row")
    ax.set_ylabel("Mean Scalar Reward")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "table_ii_csv": csv_path,
        "table_ii_json": json_path,
        "table_ii_png": plot_path,
        "table_ii_winners_csv": winners_csv_path,
        "table_ii_spreads_csv": spreads_csv_path,
        "table_ii_deltas_vs_modqn_csv": deltas_csv_path,
        "table_ii_analysis_md": analysis_md_path,
        **(
            {"long_run_reference_summary_json": reference_summary_path}
            if reference_summary_path is not None
            else {}
        ),
        "manifest": manifest_path,
    }


__all__ = [
    "export_reward_geometry_analysis",
    "export_table_ii_results",
    "export_training_run",
]
