"""Run the CP-base non-codebook continuous-power bounded matched pilot."""

from __future__ import annotations

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot import (
    run_bounded_pilot,
)


def main() -> int:
    run_bounded_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
