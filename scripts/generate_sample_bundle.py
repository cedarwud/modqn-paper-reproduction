"""Reproducibly generate the Phase 03A sample replay bundle fixture.

This script is the canonical entrypoint for refreshing
``tests/fixtures/sample-bundle-v1/``. It does NOT start a new training
experiment in the open-research sense: it runs the same one-episode
smoke training surface that is already exercised by
``tests/test_sweeps_and_export.py::test_export_cli_emits_bundle``,
exports the resulting artifact through the frozen Phase 03A producer
pipeline, then trims the timeline to a small reproducible subset that
can ship as a checked-in fixture.

Usage::

    ./.venv/bin/python scripts/generate_sample_bundle.py \
        --output tests/fixtures/sample-bundle-v1 \
        --episodes 1 --max-users 1

The resulting fixture is intentionally small (timeline rows are filtered
to the first ``max_users`` users for the first ``max_slots`` slots) so
that it can serve as a stable consumer fixture without bloating the
repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from modqn_paper_reproduction.bundle.fixture_tools import (
    sync_replay_summary_in_evaluation_summary,
)
from modqn_paper_reproduction.bundle.models import ReplaySummary
from modqn_paper_reproduction.cli import export_main, train_main
from modqn_paper_reproduction.export.replay_bundle import (
    BUNDLE_SCHEMA_VERSION,
    trim_replay_bundle_for_sample,
    validate_replay_bundle,
)


_FIXTURE_SOURCE_PLACEHOLDER = "<fixture-source-artifact-dir>"
_FIXTURE_INPUT_PLACEHOLDER = "<fixture-input-artifact-dir>"
_FIXTURE_OUTPUT_PLACEHOLDER = "<fixture-output-dir>"
_FIXTURE_CHECKPOINT_PLACEHOLDER = "<fixture-checkpoint-source>"
_FIXTURE_PRIMARY_CHECKPOINT_PLACEHOLDER = "<fixture-primary-checkpoint>"
_FIXTURE_SECONDARY_CHECKPOINT_PLACEHOLDER = "<fixture-secondary-checkpoint>"
_FIXTURE_EXPORTED_AT = "FIXTURE-DETERMINISTIC-TIMESTAMP"
_FIXTURE_ELAPSED_S = 0.0
_GITKEEP = ".gitkeep"


def _preserve_required_empty_dirs(target_dir: Path) -> None:
    """Keep schema-required empty directories present in clean checkouts."""
    sweeps_dir = target_dir / "evaluation" / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    (sweeps_dir / _GITKEEP).write_text("")


def _normalize_fixture_manifest(target_dir: Path) -> ReplaySummary | None:
    """Replace machine-specific paths/timestamps with stable placeholders.

    Returns the normalized replay summary model so the caller can mirror
    the same source object into ``evaluation/summary.json``.
    """
    manifest_path = target_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sourceArtifactDir"] = _FIXTURE_SOURCE_PLACEHOLDER
    manifest["inputArtifactDir"] = _FIXTURE_INPUT_PLACEHOLDER
    manifest["outputDir"] = _FIXTURE_OUTPUT_PLACEHOLDER
    manifest["exportedAt"] = _FIXTURE_EXPORTED_AT
    replay_summary = manifest.get("replaySummary")
    replay_summary_model = None
    if isinstance(replay_summary, dict):
        replay_summary_model = ReplaySummary.from_dict(replay_summary)
        replay_summary_model = replay_summary_model.with_checkpoint_path(
            _FIXTURE_CHECKPOINT_PLACEHOLDER
        )
        manifest["replaySummary"] = replay_summary_model.to_dict()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return replay_summary_model


def _normalize_fixture_summary(
    target_dir: Path,
    replay_summary: ReplaySummary | None,
) -> None:
    """Replace machine-specific or wall-clock fields in evaluation/summary.json.

    Also re-sync the embedded ``replay_timeline`` block from the same
    ReplaySummary source object used by ``manifest.json``.
    """
    summary_path = target_dir / "evaluation" / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    checkpoint_files = summary.get("checkpoint_files")
    if isinstance(checkpoint_files, dict):
        if "primary_final" in checkpoint_files:
            checkpoint_files["primary_final"] = _FIXTURE_PRIMARY_CHECKPOINT_PLACEHOLDER
        if "secondary_best_eval" in checkpoint_files:
            checkpoint_files["secondary_best_eval"] = (
            _FIXTURE_SECONDARY_CHECKPOINT_PLACEHOLDER
        )
    training_summary = summary.get("training_summary")
    if isinstance(training_summary, dict) and "elapsed_s" in training_summary:
        training_summary["elapsed_s"] = _FIXTURE_ELAPSED_S
    summary_path.write_text(json.dumps(summary, indent=2))
    if replay_summary is not None:
        sync_replay_summary_in_evaluation_summary(target_dir, replay_summary)


_DEFAULT_OUTPUT = "tests/fixtures/sample-bundle-v1"
_DEFAULT_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_sample_bundle")
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=(
            "Destination directory for the trimmed sample bundle "
            f"(default: {_DEFAULT_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Resolved-run config used for the smoke training pass.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of training episodes (default: 1).",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=1,
        help="Maximum users to keep in the trimmed timeline (default: 1).",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Maximum slots to keep in the trimmed timeline (default: all).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output).resolve()
    print(
        f"[generate-sample-bundle] schema={BUNDLE_SCHEMA_VERSION}",
        flush=True,
    )
    print(f"[generate-sample-bundle] target={output_dir}", flush=True)

    with tempfile.TemporaryDirectory(prefix="modqn-sample-bundle-") as scratch:
        scratch_dir = Path(scratch)
        run_dir = scratch_dir / "run"
        full_bundle_dir = scratch_dir / "full"

        train_rc = train_main(
            [
                "--config",
                args.config,
                "--episodes",
                str(args.episodes),
                "--progress-every",
                "0",
                "--output-dir",
                str(run_dir),
            ]
        )
        if train_rc != 0:
            print(
                f"[generate-sample-bundle] train_main exited with {train_rc}",
                file=sys.stderr,
            )
            return train_rc

        export_rc = export_main(
            [
                "--input",
                str(run_dir),
                "--output-dir",
                str(full_bundle_dir),
            ]
        )
        if export_rc != 0:
            print(
                f"[generate-sample-bundle] export_main exited with {export_rc}",
                file=sys.stderr,
            )
            return export_rc

        report = trim_replay_bundle_for_sample(
            full_bundle_dir,
            output_dir,
            max_users=args.max_users,
            max_slots=args.max_slots,
        )

    normalized_replay_summary = _normalize_fixture_manifest(output_dir)
    _normalize_fixture_summary(output_dir, normalized_replay_summary)
    _preserve_required_empty_dirs(output_dir)
    validate_replay_bundle(output_dir)

    print(
        "[generate-sample-bundle] kept "
        f"rows={report['rowCount']} "
        f"slots={report['slotCount']} "
        f"users={report['userCount']} "
        f"handovers={report['handoverEventCount']}",
        flush=True,
    )
    print(f"[generate-sample-bundle] manifest={report['manifestPath']}", flush=True)
    print(f"[generate-sample-bundle] timeline={report['timelinePath']}", flush=True)
    print(f"[generate-sample-bundle] note={report['sampleNote']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
