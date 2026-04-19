"""Figure-suite analysis and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._common import load_training_log_dicts, plt, write_json
from .training_log import summarize_training_log


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

    full_json_path = write_json(
        evaluation_dir / f"{suite}.json",
        {
            "rows": rows,
            "analysisContext": analysis_context,
        },
    )
    detail_csv_path = evaluation_dir / f"{suite}-detail.csv"
    weighted_csv_path = evaluation_dir / f"{suite}-weighted-reward.csv"
    winners_csv_path = evaluation_dir / f"{suite}-weighted-winners.csv"
    manifest_path = write_json(out_dir / "manifest.json", manifest)
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
        axes[-1].set_xlabel(
            f"{parameter_label} ({parameter_unit})" if parameter_unit else parameter_label
        )
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
        ax.set_xlabel(
            f"{parameter_label} ({parameter_unit})" if parameter_unit else parameter_label
        )
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
            reference_summary = summarize_training_log(load_training_log_dicts(ref_dir))
            reference_summary["reference_run_dir"] = str(ref_dir)
            reference_summary_path = evaluation_dir / f"{suite}-long-run-reference-summary.json"
            write_json(reference_summary_path, reference_summary)

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


__all__ = ["export_figure_sweep_results"]
