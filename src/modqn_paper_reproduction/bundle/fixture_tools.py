"""Fixture helpers for the Phase 04C bundle seam."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .models import ReplaySummary
from .schema import POLICY_DIAGNOSTICS_TIMELINE_FIELD
from .validator import validate_replay_bundle


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    return target


def sync_replay_summary_in_evaluation_summary(
    bundle_dir: str | Path,
    replay_summary: ReplaySummary,
) -> None:
    """Mirror one ReplaySummary source into evaluation/summary.json."""
    summary_path = Path(bundle_dir) / "evaluation" / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    summary["replay_timeline"] = replay_summary.to_dict()
    _write_json(summary_path, summary)


def trim_replay_bundle_for_sample(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    max_users: int = 1,
    max_slots: int | None = None,
    sample_note: str | None = None,
) -> dict[str, Any]:
    """Produce a small reproducible sample bundle from a full replay bundle."""
    src = Path(source_dir)
    dst = Path(target_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source bundle does not exist: {src}")

    validate_replay_bundle(src)

    if dst.is_symlink():
        raise ValueError(
            f"Refusing to overwrite symlink target for safety: {dst}. "
            "Point --target-dir at a plain directory path."
        )
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(
                f"Target path exists and is not a directory: {dst}"
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for entry in (
        "manifest.json",
        "config-resolved.json",
        "assumptions.json",
        "provenance-map.json",
        "evaluation",
        "training",
        "figures",
    ):
        source_entry = src / entry
        if not source_entry.exists():
            continue
        target_entry = dst / entry
        if source_entry.is_dir():
            shutil.copytree(source_entry, target_entry)
        else:
            shutil.copy2(source_entry, target_entry)

    timeline_src = src / "timeline" / "step-trace.jsonl"
    timeline_dst = dst / "timeline" / "step-trace.jsonl"
    timeline_dst.parent.mkdir(parents=True, exist_ok=True)

    kept_rows = 0
    kept_handovers = 0
    kept_policy_diagnostics_rows = 0
    kept_slot_indices: set[int] = set()
    kept_user_indices: set[int] = set()
    accepted_slot_set: set[int] = set()

    with timeline_src.open() as handle, timeline_dst.open("w") as out_handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            user_index = int(row.get("userIndex", -1))
            slot_index = int(row.get("slotIndex", -1))
            if user_index < 0 or user_index >= max_users:
                continue
            if max_slots is not None and slot_index not in accepted_slot_set:
                if len(accepted_slot_set) >= max_slots:
                    continue
                accepted_slot_set.add(slot_index)
            out_handle.write(json.dumps(row) + "\n")
            kept_rows += 1
            kept_slot_indices.add(slot_index)
            kept_user_indices.add(user_index)
            if POLICY_DIAGNOSTICS_TIMELINE_FIELD in row:
                kept_policy_diagnostics_rows += 1
            handover_event = row.get("handoverEvent") or {}
            if handover_event.get("eventId"):
                kept_handovers += 1

    if kept_rows == 0:
        raise ValueError(
            "Trimmed replay bundle would be empty. "
            "Check max_users/max_slots against the source timeline."
        )

    manifest_path = dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    replay_summary = ReplaySummary.from_dict(manifest["replaySummary"])

    full_row_count = int(replay_summary.row_count)
    full_slot_count = int(replay_summary.slot_count)
    full_handover_count = int(replay_summary.handover_event_count)

    sample_note_text = sample_note or (
        "Trimmed for fixture/sample purposes. Timeline reduced to the first "
        f"{len(kept_user_indices)} user(s) and first {len(kept_slot_indices)} "
        "slot(s) of the source bundle. All other surfaces are byte-equal to "
        "the source."
    )

    replay_summary = replay_summary.with_trimmed_subset(
        row_count=kept_rows,
        slot_count=len(kept_slot_indices),
        handover_event_count=kept_handovers,
        max_users=max_users,
        max_slots=max_slots,
        user_indices=sorted(kept_user_indices),
        slot_indices=sorted(kept_slot_indices),
        source_full_row_count=full_row_count,
        source_full_slot_count=full_slot_count,
        source_full_handover_event_count=full_handover_count,
    )
    manifest["replaySummary"] = replay_summary.to_dict()
    if "outputDir" in manifest:
        manifest["outputDir"] = str(dst)

    optional_policy_diagnostics = manifest.get("optionalPolicyDiagnostics")
    if isinstance(optional_policy_diagnostics, dict):
        optional_policy_diagnostics["present"] = bool(kept_policy_diagnostics_rows > 0)
        optional_policy_diagnostics["rowsWithDiagnostics"] = int(
            kept_policy_diagnostics_rows
        )
        optional_policy_diagnostics["rowsWithoutDiagnostics"] = int(
            kept_rows - kept_policy_diagnostics_rows
        )
    manifest["sampleNote"] = sample_note_text
    _write_json(manifest_path, manifest)
    sync_replay_summary_in_evaluation_summary(dst, replay_summary)

    validate_replay_bundle(dst)

    return {
        "targetDir": dst,
        "rowCount": kept_rows,
        "slotCount": len(kept_slot_indices),
        "userCount": len(kept_user_indices),
        "handoverEventCount": kept_handovers,
        "manifestPath": manifest_path,
        "timelinePath": timeline_dst,
        "sampleNote": sample_note_text,
    }
