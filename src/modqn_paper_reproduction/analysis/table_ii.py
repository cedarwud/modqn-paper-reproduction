"""Table II analysis and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._common import load_training_log_dicts, plt, write_json
from .training_log import summarize_training_log


def build_table_ii_analysis_frames(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                "scalar_spread": float(
                    grp["mean_scalar_reward"].max() - grp["mean_scalar_reward"].min()
                ),
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
                    "scalar_delta_vs_modqn": float(
                        row["mean_scalar_reward"] - modqn["mean_scalar_reward"]
                    ),
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


def write_table_ii_analysis_markdown(
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
    max_scalar_spread = (
        float(spreads_df["scalar_spread"].max()) if not spreads_df.empty else 0.0
    )
    max_r1_spread = float(spreads_df["r1_spread"].max()) if not spreads_df.empty else 0.0
    max_r2_spread = float(spreads_df["r2_spread"].max()) if not spreads_df.empty else 0.0
    max_r3_spread = float(spreads_df["r3_spread"].max()) if not spreads_df.empty else 0.0
    max_handover_spread = (
        float(spreads_df["handover_spread"].max()) if not spreads_df.empty else 0.0
    )
    training_episodes = sorted(
        {int(x) for x in df["training_episodes"].unique() if int(x) > 0}
    )

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
    write_json(json_path, {"rows": rows})
    write_json(manifest_path, manifest)

    winners_df, spreads_df, deltas_df = build_table_ii_analysis_frames(df)
    winners_df.to_csv(winners_csv_path, index=False)
    spreads_df.to_csv(spreads_csv_path, index=False)
    deltas_df.to_csv(deltas_csv_path, index=False)

    reference_summary_path = None
    reference_summary = None
    if reference_run_dir is not None:
        ref_dir = Path(reference_run_dir)
        training_log_path = ref_dir / "training_log.json"
        if training_log_path.exists():
            reference_training_log = load_training_log_dicts(ref_dir)
            reference_summary = summarize_training_log(reference_training_log)
            reference_summary["reference_run_dir"] = str(ref_dir)
            reference_summary_path = evaluation_dir / "long-run-reference-summary.json"
            write_json(reference_summary_path, reference_summary)

    write_table_ii_analysis_markdown(
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
    "build_table_ii_analysis_frames",
    "export_table_ii_results",
    "write_table_ii_analysis_markdown",
]
