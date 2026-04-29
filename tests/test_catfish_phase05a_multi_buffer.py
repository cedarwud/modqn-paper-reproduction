from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from modqn_paper_reproduction.analysis.catfish_phase05a_multi_buffer import (
    PHASE_05A_ANALYSIS_KIND,
    SCALAR_BUFFER_NAME,
    Phase05ASample,
    analyze_phase05a_samples,
    jaccard,
    objective_scores,
    validate_phase05a_analysis_config,
)
from modqn_paper_reproduction.cli import catfish_phase05a_multi_buffer_main
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_trainer_config,
    load_training_yaml,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
PHASE05A_CONFIG = "configs/catfish-modqn-phase-05a-multi-buffer-primary.resolved.yaml"


def test_objective_ranking_treats_r2_and_r3_closer_to_zero_as_better() -> None:
    rewards = np.array(
        [
            [10.0, -0.50, -2.00],
            [2.0, -0.05, -0.20],
            [1.0, 0.00, 0.00],
        ],
        dtype=np.float64,
    )

    assert objective_scores(rewards, "r1").tolist() == [10.0, 2.0, 1.0]
    assert objective_scores(rewards, "r2")[2] > objective_scores(rewards, "r2")[1]
    assert objective_scores(rewards, "r2")[1] > objective_scores(rewards, "r2")[0]
    assert objective_scores(rewards, "r3")[2] > objective_scores(rewards, "r3")[1]
    assert objective_scores(rewards, "r3")[1] > objective_scores(rewards, "r3")[0]


def test_r1_r2_r3_buffers_are_selected_independently() -> None:
    diagnostics = analyze_phase05a_samples(_independent_samples(), objective_top_quantile=0.8)

    assert diagnostics["buffers"]["r1"]["size"] == 2
    assert diagnostics["buffers"]["r2"]["size"] == 2
    assert diagnostics["buffers"]["r3"]["size"] == 2
    assert diagnostics["overlap"]["objective_jaccard"]["r1_vs_r2"] == 0.0
    assert diagnostics["overlap"]["objective_jaccard"]["r1_vs_r3"] == 0.0
    assert diagnostics["overlap"]["objective_jaccard"]["r2_vs_r3"] == 0.0
    assert (
        diagnostics["buffers"]["r2"]["distinct_sample_contribution"][
            "would_contribute_distinct_intervention_samples"
        ]
        is True
    )
    assert (
        diagnostics["buffers"]["r3"]["distinct_sample_contribution"][
            "would_contribute_distinct_intervention_samples"
        ]
        is True
    )


def test_jaccard_overlap_is_computed_correctly() -> None:
    assert jaccard({1, 2, 3}, {3, 4}) == pytest.approx(0.25)
    assert jaccard(set(), set()) == pytest.approx(1.0)


def test_scalar_high_value_buffer_is_not_objective_specific_evidence() -> None:
    diagnostics = analyze_phase05a_samples(_independent_samples(), objective_top_quantile=0.8)

    assert SCALAR_BUFFER_NAME not in diagnostics["buffers"]
    assert SCALAR_BUFFER_NAME in diagnostics["sample_admission_counts"]
    assert diagnostics["claim_boundary"]["scalar_reward_alone_used_as_evidence"] is False
    assert "scalar reward is not used" in diagnostics["buffers"]["r2"]["selection_rule"]


def test_phase05a_config_validates_but_rejects_ee_and_full_multi_catfish() -> None:
    cfg = load_training_yaml(PHASE05A_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    phase_block = validate_phase05a_analysis_config(cfg, trainer_cfg)

    assert cfg["analysis_experiment"]["kind"] == PHASE_05A_ANALYSIS_KIND
    assert phase_block["full_multi_catfish_agents"] == "disabled"
    assert trainer_cfg.r1_reward_mode == "throughput"
    assert trainer_cfg.method_family == "Catfish-MODQN"

    ee_cfg = copy.deepcopy(cfg)
    ee_cfg["analysis_experiment"]["phase_05a_multi_buffer"]["ee_reward"] = "enabled"
    with pytest.raises(ConfigValidationError):
        validate_phase05a_analysis_config(ee_cfg, trainer_cfg)

    multi_cfg = copy.deepcopy(cfg)
    multi_cfg["analysis_experiment"]["phase_05a_multi_buffer"][
        "full_multi_catfish_agents"
    ] = "enabled"
    with pytest.raises(ConfigValidationError):
        validate_phase05a_analysis_config(multi_cfg, trainer_cfg)


def test_phase05a_command_rejects_full_multi_catfish_config(tmp_path: Path) -> None:
    cfg = load_training_yaml(PHASE05A_CONFIG)
    cfg["inherits_from"] = str(Path("configs/catfish-modqn-phase-04-b-primary-shaping-off.resolved.yaml").resolve())
    cfg["analysis_experiment"]["phase_05a_multi_buffer"][
        "full_multi_catfish_agents"
    ] = "enabled"
    bad_config = tmp_path / "bad-phase05a.yaml"
    bad_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rc = catfish_phase05a_multi_buffer_main(
        [
            "--config",
            str(bad_config),
            "--output-dir",
            str(tmp_path / "out"),
            "--episodes",
            "1",
        ]
    )

    assert rc == 2
    assert not (tmp_path / "out" / "phase05a_multi_buffer_diagnostics.json").exists()


def test_baseline_behavior_remains_unchanged() -> None:
    trainer_cfg = build_trainer_config(load_training_yaml(BASELINE_CONFIG))

    assert trainer_cfg.training_experiment_kind == "baseline"
    assert trainer_cfg.method_family == "MODQN-baseline"
    assert trainer_cfg.r1_reward_mode == "throughput"
    assert trainer_cfg.catfish_enabled is False
    assert trainer_cfg.catfish_intervention_enabled is False


def _independent_samples() -> list[Phase05ASample]:
    rows = [
        (10.0, -5.0, -5.0, True),
        (9.0, -5.0, -5.0, True),
        (1.0, 0.0, -5.0, False),
        (1.0, -0.05, -5.0, False),
        (1.0, -5.0, 0.0, False),
        (1.0, -5.0, -0.05, False),
        (1.0, -5.0, -5.0, False),
        (1.0, -5.0, -5.0, False),
        (2.0, -4.0, -4.0, False),
        (1.0, -4.0, -4.0, False),
    ]
    samples: list[Phase05ASample] = []
    for idx, (r1, r2, r3, admitted) in enumerate(rows):
        samples.append(
            Phase05ASample(
                sample_id=idx,
                r1=r1,
                r2=r2,
                r3=r3,
                scalar_quality=0.5 * r1 + 0.3 * r2 + 0.2 * r3,
                scalar_phase04_admitted=admitted,
            )
        )
    return samples
