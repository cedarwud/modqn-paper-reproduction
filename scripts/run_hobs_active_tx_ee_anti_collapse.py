"""Run the bounded HOBS active-TX EE anti-collapse design gate."""

from __future__ import annotations

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_anti_collapse import (
    export_tiny_matched_anti_collapse_pilot,
)


def main() -> int:
    export_tiny_matched_anti_collapse_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
