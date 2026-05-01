"""Run the bounded QoS-sticky HOBS active-TX EE anti-collapse gate."""

from __future__ import annotations

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_qos_sticky_anti_collapse import (
    export_tiny_matched_qos_sticky_anti_collapse_pilot,
)


def main() -> int:
    export_tiny_matched_qos_sticky_anti_collapse_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
