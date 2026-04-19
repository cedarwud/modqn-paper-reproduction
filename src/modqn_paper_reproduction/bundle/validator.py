"""Bundle validators split out in Phase 04C."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    OPTIONAL_POLICY_DIAGNOSTICS_REQUIRED_FIELDS,
    POLICY_DIAGNOSTICS_REQUIRED_FIELDS,
    POLICY_DIAGNOSTICS_TIMELINE_FIELD,
    POLICY_DIAGNOSTICS_TOP_CANDIDATE_REQUIRED_FIELDS,
    REQUIRED_BUNDLE_RELATIVE_PATHS,
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_NON_EMPTY_MANIFEST_FIELDS,
    TIMELINE_ROW_REQUIRED_FIELDS,
)


def _validate_objective_triplet_mapping(
    payload: Any,
    *,
    field_name: str,
) -> None:
    expected_keys = {"r1Throughput", "r2Handover", "r3LoadBalance"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            f"{field_name} must be an object with keys {sorted(expected_keys)}."
        )
    for key, value in payload.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_name}.{key} must be numeric.")


def _validate_policy_diagnostics_row(
    row: dict[str, Any],
    *,
    line_number: int,
) -> bool:
    payload = row.get(POLICY_DIAGNOSTICS_TIMELINE_FIELD)
    if payload is None:
        return False

    missing = [
        field for field in POLICY_DIAGNOSTICS_REQUIRED_FIELDS if field not in payload
    ]
    if missing:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics is missing required fields: {missing}"
        )

    if not payload["diagnosticsVersion"]:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.diagnosticsVersion must be non-empty."
        )

    _validate_objective_triplet_mapping(
        payload["objectiveWeights"],
        field_name=(
            f"Replay timeline row {line_number} policyDiagnostics.objectiveWeights"
        ),
    )
    if not isinstance(payload["selectedScalarizedQ"], (int, float)) or isinstance(
        payload["selectedScalarizedQ"], bool
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.selectedScalarizedQ must be numeric."
        )
    if payload["runnerUpScalarizedQ"] is not None and (
        not isinstance(payload["runnerUpScalarizedQ"], (int, float))
        or isinstance(payload["runnerUpScalarizedQ"], bool)
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.runnerUpScalarizedQ must be numeric or null."
        )
    if payload["scalarizedMarginToRunnerUp"] is not None and (
        not isinstance(payload["scalarizedMarginToRunnerUp"], (int, float))
        or isinstance(payload["scalarizedMarginToRunnerUp"], bool)
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.scalarizedMarginToRunnerUp must be numeric or null."
        )
    if not isinstance(payload["availableActionCount"], int) or isinstance(
        payload["availableActionCount"], bool
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.availableActionCount must be an integer."
        )

    top_candidates = payload["topCandidates"]
    if not isinstance(top_candidates, list) or not top_candidates:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.topCandidates must be a non-empty list."
        )

    selected_serving = row.get("selectedServing") or {}
    selected_beam_index = int(selected_serving.get("beamIndex", -1))
    decision_mask = row.get("decisionActionValidityMask")
    if not isinstance(decision_mask, list):
        decision_mask = row.get("actionValidityMask")
    available_action_count = None
    if isinstance(decision_mask, list):
        available_action_count = sum(bool(value) for value in decision_mask)
        if available_action_count != int(payload["availableActionCount"]):
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics.availableActionCount does not match the decision mask."
            )

    previous_scalarized = None
    previous_beam_index = None
    for candidate_index, candidate in enumerate(top_candidates, start=1):
        missing_candidate = [
            field
            for field in POLICY_DIAGNOSTICS_TOP_CANDIDATE_REQUIRED_FIELDS
            if field not in candidate
        ]
        if missing_candidate:
            raise ValueError(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}] "
                f"is missing required fields: {missing_candidate}"
            )
        if not bool(candidate["validUnderDecisionMask"]):
            raise ValueError(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}] "
                "must remain valid under the decision mask."
            )
        beam_index = int(candidate["beamIndex"])
        if isinstance(decision_mask, list):
            if beam_index < 0 or beam_index >= len(decision_mask) or not bool(
                decision_mask[beam_index]
            ):
                raise ValueError(
                    "Replay timeline row "
                    f"{line_number} policyDiagnostics.topCandidates[{candidate_index}] "
                    "does not respect decisionActionValidityMask."
                )
        _validate_objective_triplet_mapping(
            candidate["objectiveQ"],
            field_name=(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}].objectiveQ"
            ),
        )
        scalarized_q = candidate["scalarizedQ"]
        if not isinstance(scalarized_q, (int, float)) or isinstance(
            scalarized_q, bool
        ):
            raise ValueError(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}].scalarizedQ "
                "must be numeric."
            )
        scalarized_q = float(scalarized_q)
        if previous_scalarized is not None:
            if scalarized_q > previous_scalarized + 1e-9:
                raise ValueError(
                    f"Replay timeline row {line_number} policyDiagnostics.topCandidates must be sorted by descending scalarizedQ."
                )
            if (
                abs(scalarized_q - previous_scalarized) <= 1e-9
                and previous_beam_index is not None
                and beam_index < previous_beam_index
            ):
                raise ValueError(
                    f"Replay timeline row {line_number} policyDiagnostics.topCandidates must use ascending beamIndex tie-breaks."
                )
        previous_scalarized = scalarized_q
        previous_beam_index = beam_index

    selected_candidate = top_candidates[0]
    if int(selected_candidate["beamIndex"]) != selected_beam_index:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics must align topCandidates[0] with selectedServing."
        )
    if (
        abs(
            float(selected_candidate["scalarizedQ"])
            - float(payload["selectedScalarizedQ"])
        )
        > 1e-9
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.selectedScalarizedQ must match topCandidates[0].scalarizedQ."
        )

    has_runner_up = len(top_candidates) > 1
    if has_runner_up:
        runner_up_scalarized = payload["runnerUpScalarizedQ"]
        margin = payload["scalarizedMarginToRunnerUp"]
        if runner_up_scalarized is None or margin is None:
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics must include runner-up score and margin when more than one action is available."
            )
        if (
            abs(float(top_candidates[1]["scalarizedQ"]) - float(runner_up_scalarized))
            > 1e-9
        ):
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics.runnerUpScalarizedQ must match topCandidates[1].scalarizedQ."
            )
        expected_margin = float(payload["selectedScalarizedQ"]) - float(
            runner_up_scalarized
        )
        if abs(float(margin) - expected_margin) > 1e-9:
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics.scalarizedMarginToRunnerUp is inconsistent with selected/runner-up scores."
            )
    else:
        if (
            payload["runnerUpScalarizedQ"] is not None
            or payload["scalarizedMarginToRunnerUp"] is not None
        ):
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics must keep runner-up score and margin null when only one candidate is available."
            )

    return True


def _validate_optional_policy_diagnostics_manifest(
    manifest: dict[str, Any],
    *,
    timeline_row_count: int,
    rows_with_diagnostics: int,
) -> None:
    payload = manifest.get("optionalPolicyDiagnostics")
    if payload is None:
        if rows_with_diagnostics > 0:
            raise ValueError(
                "Replay bundle timeline exports policyDiagnostics rows but manifest.optionalPolicyDiagnostics is missing."
            )
        return

    missing = [
        field
        for field in OPTIONAL_POLICY_DIAGNOSTICS_REQUIRED_FIELDS
        if field not in payload
    ]
    if missing:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics is missing required fields: "
            f"{missing}"
        )
    if payload["timelineField"] != POLICY_DIAGNOSTICS_TIMELINE_FIELD:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.timelineField must be "
            f"{POLICY_DIAGNOSTICS_TIMELINE_FIELD!r}."
        )
    if not payload["diagnosticsVersion"]:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.diagnosticsVersion must be non-empty."
        )
    if int(payload["rowsWithDiagnostics"]) != rows_with_diagnostics:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.rowsWithDiagnostics does not match the timeline."
        )
    rows_without_diagnostics = timeline_row_count - rows_with_diagnostics
    if int(payload["rowsWithoutDiagnostics"]) != rows_without_diagnostics:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.rowsWithoutDiagnostics does not match the timeline."
        )
    if bool(payload["present"]) != bool(rows_with_diagnostics > 0):
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.present does not match timeline coverage."
        )


def validate_replay_bundle(output_dir: str | Path) -> None:
    """Validate the minimum replay-complete Phase 03A producer bundle surface."""
    out_dir = Path(output_dir)
    required_paths = tuple(out_dir / relative for relative in REQUIRED_BUNDLE_RELATIVE_PATHS)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(f"Replay bundle is missing required paths: {missing}")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    missing_manifest_fields = [
        field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest
    ]
    if missing_manifest_fields:
        raise ValueError(
            "Replay bundle manifest is missing required fields: "
            f"{missing_manifest_fields}"
        )
    empty_identity_fields = [
        field
        for field in REQUIRED_NON_EMPTY_MANIFEST_FIELDS
        if manifest.get(field) in (None, "")
    ]
    if empty_identity_fields:
        raise ValueError(
            "Replay bundle manifest has empty identity fields: "
            f"{empty_identity_fields}"
        )

    timeline_path = out_dir / "timeline" / "step-trace.jsonl"
    row_count = 0
    rows_with_policy_diagnostics = 0
    with timeline_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Replay timeline row {line_number} is not valid JSON: {exc}"
                ) from exc
            missing_row_fields = [
                field for field in TIMELINE_ROW_REQUIRED_FIELDS if field not in row
            ]
            if missing_row_fields:
                raise ValueError(
                    f"Replay timeline row {line_number} is missing required fields: "
                    f"{missing_row_fields}"
                )
            if _validate_policy_diagnostics_row(row, line_number=line_number):
                rows_with_policy_diagnostics += 1
            row_count += 1
    if row_count == 0:
        raise ValueError("Replay bundle timeline is empty.")
    _validate_optional_policy_diagnostics_manifest(
        manifest,
        timeline_row_count=row_count,
        rows_with_diagnostics=rows_with_policy_diagnostics,
    )
