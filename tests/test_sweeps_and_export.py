from __future__ import annotations

import csv
import json
from pathlib import Path

from modqn_paper_reproduction.algorithms import ScalarDQNPolicyConfig, ScalarDQNTrainer
from modqn_paper_reproduction.algorithms.modqn import TrainerConfig
from modqn_paper_reproduction.baselines import evaluate_rss_max
from modqn_paper_reproduction.cli import export_main, sweep_main, train_main
from modqn_paper_reproduction.config_loader import build_environment, get_seeds, load_training_yaml


RESOLVED_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"


def test_scalar_dqn_comparator_smoke() -> None:
    cfg = load_training_yaml(RESOLVED_CONFIG)
    env = build_environment(cfg)
    seeds = get_seeds(cfg)
    tc = TrainerConfig(
        episodes=2,
        hidden_layers=(32, 16),
        batch_size=32,
        replay_capacity=512,
        target_update_every_episodes=1,
    )
    trainer = ScalarDQNTrainer(
        env=env,
        config=tc,
        policy=ScalarDQNPolicyConfig(
            name="DQN_throughput",
            scalar_reward_weights=(1.0, 0.0, 0.0),
        ),
        train_seed=seeds["train_seed"],
        env_seed=seeds["environment_seed"],
        mobility_seed=seeds["mobility_seed"],
    )
    logs = trainer.train(
        progress_every=0,
        evaluation_seed_set=(100, 200),
        evaluation_every_episodes=1,
    )
    assert len(logs) == 2
    assert trainer.has_best_eval_checkpoint()
    trainer.restore_best_eval_checkpoint()
    summary = trainer.evaluate_policy(
        (100, 200),
        episode=1,
        evaluation_every_episodes=1,
        scalarization_weights=(0.5, 0.3, 0.2),
    )
    assert summary.mean_r1 > 0.0
    assert len(summary.eval_seeds) == 2


def test_rss_max_eval_smoke() -> None:
    cfg = load_training_yaml(RESOLVED_CONFIG)
    env = build_environment(cfg)
    summary = evaluate_rss_max(
        env,
        evaluation_seed_set=(100, 200),
        scalarization_weights=(0.5, 0.3, 0.2),
    )
    assert summary.mean_r1 > 0.0
    assert summary.mean_total_handovers >= 0.0


def test_sweep_cli_table_ii_outputs(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference-run"
    train_rc = train_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--episodes",
            "2",
            "--progress-every",
            "0",
            "--output-dir",
            str(reference_run),
        ]
    )
    assert train_rc == 0

    out_dir = tmp_path / "table-ii"
    rc = sweep_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--suite",
            "table-ii",
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--max-weight-rows",
            "2",
            "--output-dir",
            str(out_dir),
            "--reference-run",
            str(reference_run),
        ]
    )
    assert rc == 0

    csv_path = out_dir / "figures" / "table-ii.csv"
    json_path = out_dir / "evaluation" / "table-ii.json"
    png_path = out_dir / "figures" / "table-ii.png"
    manifest_path = out_dir / "manifest.json"
    winners_csv_path = out_dir / "evaluation" / "table-ii-winners.csv"
    spreads_csv_path = out_dir / "evaluation" / "table-ii-spreads.csv"
    deltas_csv_path = out_dir / "evaluation" / "table-ii-deltas-vs-modqn.csv"
    reference_summary_path = out_dir / "evaluation" / "long-run-reference-summary.json"
    analysis_md_path = out_dir / "analysis" / "table-ii-analysis.md"

    assert csv_path.exists()
    assert json_path.exists()
    assert png_path.exists()
    assert manifest_path.exists()
    assert winners_csv_path.exists()
    assert spreads_csv_path.exists()
    assert deltas_csv_path.exists()
    assert reference_summary_path.exists()
    assert analysis_md_path.exists()

    rows = list(csv.DictReader(csv_path.open()))
    assert rows
    assert {"MODQN", "DQN_throughput", "DQN_scalar", "RSS_max"} <= {
        row["method"] for row in rows
    }

    manifest = json.loads(manifest_path.read_text())
    assert manifest["paperId"] == "PAP-2024-MORL-MULTIBEAM"

    winners = list(csv.DictReader(winners_csv_path.open()))
    assert winners
    spreads = list(csv.DictReader(spreads_csv_path.open()))
    assert spreads
    deltas = list(csv.DictReader(deltas_csv_path.open()))
    assert deltas
    reference_summary = json.loads(reference_summary_path.read_text())
    assert reference_summary["reference_run_dir"] == str(reference_run)
    analysis_text = analysis_md_path.read_text()
    assert "Long-Run Reference" in analysis_text


def test_sweep_cli_reward_geometry_outputs(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference-run"
    train_rc = train_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--episodes",
            "3",
            "--progress-every",
            "0",
            "--output-dir",
            str(reference_run),
        ]
    )
    assert train_rc == 0

    table_dir = tmp_path / "table-ii"
    table_rc = sweep_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--suite",
            "table-ii",
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--max-weight-rows",
            "2",
            "--output-dir",
            str(table_dir),
            "--reference-run",
            str(reference_run),
        ]
    )
    assert table_rc == 0

    out_dir = tmp_path / "reward-geometry"
    rc = sweep_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--suite",
            "reward-geometry",
            "--input-table-ii",
            str(table_dir),
            "--reference-run",
            str(reference_run),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    assert (out_dir / "evaluation" / "reward-geometry-diagnostics.json").exists()
    assert (out_dir / "evaluation" / "reward-geometry-scales.csv").exists()
    assert (out_dir / "evaluation" / "reward-geometry-table-ii.csv").exists()
    assert (out_dir / "evaluation" / "reward-geometry-winners.csv").exists()
    assert (out_dir / "evaluation" / "reward-geometry-reference-window-contributions.csv").exists()
    assert (out_dir / "evaluation" / "reward-geometry-summary.json").exists()
    assert (out_dir / "figures" / "reward-geometry-scalar-spread.png").exists()
    analysis_text = (out_dir / "analysis" / "reward-geometry.md").read_text()
    assert "experimental" in analysis_text.lower()


def test_sweep_cli_fig3_outputs(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference-run"
    train_rc = train_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--episodes",
            "2",
            "--progress-every",
            "0",
            "--output-dir",
            str(reference_run),
        ]
    )
    assert train_rc == 0

    out_dir = tmp_path / "fig-3"
    rc = sweep_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--suite",
            "fig-3",
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--max-figure-points",
            "2",
            "--output-dir",
            str(out_dir),
            "--reference-run",
            str(reference_run),
        ]
    )
    assert rc == 0

    detail_csv_path = out_dir / "evaluation" / "fig-3-detail.csv"
    weighted_csv_path = out_dir / "evaluation" / "fig-3-weighted-reward.csv"
    winners_csv_path = out_dir / "evaluation" / "fig-3-weighted-winners.csv"
    json_path = out_dir / "evaluation" / "fig-3.json"
    objectives_png_path = out_dir / "figures" / "fig-3-objectives.png"
    weighted_png_path = out_dir / "figures" / "fig-3-weighted-reward.png"
    reference_summary_path = out_dir / "evaluation" / "fig-3-long-run-reference-summary.json"
    analysis_md_path = out_dir / "analysis" / "fig-3-analysis.md"
    manifest_path = out_dir / "manifest.json"

    assert detail_csv_path.exists()
    assert weighted_csv_path.exists()
    assert winners_csv_path.exists()
    assert json_path.exists()
    assert objectives_png_path.exists()
    assert weighted_png_path.exists()
    assert reference_summary_path.exists()
    assert analysis_md_path.exists()
    assert manifest_path.exists()

    rows = list(csv.DictReader(detail_csv_path.open()))
    assert rows
    assert {"MODQN", "DQN_throughput", "DQN_scalar", "RSS_max"} <= {
        row["method"] for row in rows
    }
    assert {40.0, 60.0} == {float(row["parameter_value"]) for row in rows}

    manifest = json.loads(manifest_path.read_text())
    assert manifest["figureId"] == "Fig. 3"
    assert manifest["sweepParameter"] == "user_count"

    analysis_text = analysis_md_path.read_text()
    assert "weighted reward" in analysis_text.lower()


def test_sweep_cli_fig4_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "fig-4"
    rc = sweep_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--suite",
            "fig-4",
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--max-figure-points",
            "2",
            "--methods",
            "modqn,rss_max",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    detail_rows = list(csv.DictReader((out_dir / "evaluation" / "fig-4-detail.csv").open()))
    assert detail_rows
    assert {2.0, 3.0} == {float(row["parameter_value"]) for row in detail_rows}
    assert {"MODQN", "RSS_max"} == {row["method"] for row in detail_rows}

    winners_rows = list(csv.DictReader((out_dir / "evaluation" / "fig-4-weighted-winners.csv").open()))
    assert winners_rows
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["figureId"] == "Fig. 4"
    assert manifest["sweepParameter"] == "satellite_count"


def test_export_cli_emits_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    export_dir = tmp_path / "export"

    train_rc = train_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert train_rc == 0

    export_rc = export_main(
        [
            "--input",
            str(run_dir),
            "--output-dir",
            str(export_dir),
        ]
    )
    assert export_rc == 0
    assert (export_dir / "manifest.json").exists()
    assert (export_dir / "assumptions.json").exists()
    assert (export_dir / "training" / "episode_metrics.csv").exists()
    assert (export_dir / "training" / "loss_curves.csv").exists()
    assert (export_dir / "figures" / "training-scalar-reward.png").exists()
    assert (export_dir / "figures" / "training-objectives.png").exists()
