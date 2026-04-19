"""Reward-geometry analysis helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._common import load_training_log_dicts, plt, safe_abs_scale, write_json
from .training_log import summarize_training_log


def collect_reward_diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
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


def build_reward_geometry_scale_table(
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
            "scale_r1": safe_abs_scale(diagnostics["sample_r1"]),
            "scale_r2": safe_abs_scale(diagnostics["sample_r2_beam_change"]),
            "scale_r3": safe_abs_scale(diagnostics["sample_r3"]),
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
                    "scale_r1": safe_abs_scale(first["r1_mean"]),
                    "scale_r2": safe_abs_scale(first["r2_mean"]),
                    "scale_r3": safe_abs_scale(first["r3_mean"]),
                }
            )
        if last:
            rows.append(
                {
                    "scenario": "reference-last-window-abs",
                    "source": "reference-run:last-window",
                    "scale_r1": safe_abs_scale(last["r1_mean"]),
                    "scale_r2": safe_abs_scale(last["r2_mean"]),
                    "scale_r3": safe_abs_scale(last["r3_mean"]),
                }
            )
    return pd.DataFrame(rows)


def build_reward_geometry_table_ii_frames(
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
            reference_summary = summarize_training_log(load_training_log_dicts(ref_dir))
            reference_summary["reference_run_dir"] = str(ref_dir)

    diagnostics = collect_reward_diagnostics(cfg)
    scale_df = build_reward_geometry_scale_table(diagnostics, reference_summary)
    scenario_df, winner_df = build_reward_geometry_table_ii_frames(table_df, scale_df)

    baseline_weights = tuple(
        float(x)
        for x in cfg.get("baseline", cfg).get("objective_weights", [0.5, 0.3, 0.2])
    )
    reference_window_df = _build_reference_window_contributions(
        reference_summary,
        scale_df,
        baseline_weights=baseline_weights,
    )

    diagnostics_path = write_json(
        evaluation_dir / "reward-geometry-diagnostics.json",
        diagnostics,
    )
    scales_csv_path = evaluation_dir / "reward-geometry-scales.csv"
    scale_df.to_csv(scales_csv_path, index=False)
    scenario_csv_path = evaluation_dir / "reward-geometry-table-ii.csv"
    scenario_df.to_csv(scenario_csv_path, index=False)
    winner_csv_path = evaluation_dir / "reward-geometry-winners.csv"
    winner_df.to_csv(winner_csv_path, index=False)
    reference_window_csv_path = (
        evaluation_dir / "reward-geometry-reference-window-contributions.csv"
    )
    if reference_window_df.empty:
        pd.DataFrame(
            columns=[
                "window",
                "scenario",
                "scalar",
                "contribution_r1",
                "contribution_r2",
                "contribution_r3",
            ]
        ).to_csv(reference_window_csv_path, index=False)
    else:
        reference_window_df.to_csv(reference_window_csv_path, index=False)

    summary_path = write_json(
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


__all__ = [
    "build_reward_geometry_scale_table",
    "build_reward_geometry_table_ii_frames",
    "collect_reward_diagnostics",
    "export_reward_geometry_analysis",
]
