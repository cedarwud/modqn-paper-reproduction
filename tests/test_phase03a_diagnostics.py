from __future__ import annotations

import numpy as np

from modqn_paper_reproduction.analysis.phase03a_diagnostics import (
    concentration_metrics,
    distribution,
    select_counterfactual_actions,
)
from modqn_paper_reproduction.env.step import ActionMask


def _mask(values: list[int]) -> ActionMask:
    return ActionMask(mask=np.asarray(values, dtype=bool))


def test_concentration_metrics_identifies_single_bucket_collapse() -> None:
    metrics = concentration_metrics([100, 0, 0, 0])

    assert metrics["active_count"] == 1
    assert metrics["entropy_normalized"] == 0.0
    assert metrics["hhi"] == 1.0
    assert metrics["top1_share"] == 1.0
    assert metrics["all_mass_one_bucket"] is True


def test_concentration_metrics_reports_even_spread() -> None:
    metrics = concentration_metrics([25, 25, 25, 25])

    assert metrics["active_count"] == 4
    assert np.isclose(metrics["entropy_normalized"], 1.0)
    assert np.isclose(metrics["hhi"], 0.25)
    assert np.isclose(metrics["top1_share"], 0.25)
    assert metrics["all_mass_one_bucket"] is False


def test_counterfactual_spread_valid_uses_distinct_valid_actions() -> None:
    masks = [_mask([1, 1, 1, 0]) for _ in range(6)]
    actions, diagnostics = select_counterfactual_actions(
        "spread-valid-heuristic",
        current_assignments=np.zeros(6, dtype=np.int32),
        masks=masks,
        rng=np.random.default_rng(123),
    )

    assert actions.tolist() == [0, 1, 2, 0, 1, 2]
    assert {row["selectionRule"] for row in diagnostics} == {
        "least-assigned-valid-then-lowest-index"
    }


def test_counterfactual_hold_current_falls_back_when_current_is_invalid() -> None:
    actions, diagnostics = select_counterfactual_actions(
        "hold-current",
        current_assignments=np.asarray([2, 3], dtype=np.int32),
        masks=[_mask([1, 1, 1, 0]), _mask([0, 1, 0, 0])],
        rng=np.random.default_rng(123),
    )

    assert actions.tolist() == [2, 1]
    assert diagnostics[0]["fallbackUsed"] is False
    assert diagnostics[1]["fallbackUsed"] is True


def test_distribution_keeps_distinct_values_and_histogram() -> None:
    stats = distribution([2.0, 2.0, 10.0])

    assert stats["count"] == 3
    assert stats["distinct"] == [2.0, 10.0]
    assert stats["histogram"] == {"2.0": 2, "10.0": 1}
