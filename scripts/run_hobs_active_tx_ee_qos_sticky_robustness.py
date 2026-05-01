"""Run the bounded QoS-sticky HOBS active-TX EE robustness gate."""

from __future__ import annotations

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_qos_sticky_robustness import (
    run_qos_sticky_robustness_gate,
)


def main() -> int:
    run_qos_sticky_robustness_gate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
