"""Bundle provenance helpers split out in Phase 04C."""

from __future__ import annotations

from typing import Any

from .. import PACKAGE_VERSION
from ..artifacts import RunMetadataV1
from .schema import BUNDLE_SCHEMA_VERSION


_PAPER_BACKED_BASELINE_FIELDS: dict[str, str] = {
    "satellites": "Table I baseline topology count.",
    "beams_per_satellite": "Table I baseline beam count.",
    "users": "Baseline user count used by the paper protocol.",
    "altitude_km": "Table I satellite altitude.",
    "user_speed_kmh": "Figure sweep baseline user speed.",
    "satellite_speed_km_s": "Figure sweep baseline satellite speed.",
    "carrier_frequency_ghz": "Table I carrier frequency.",
    "bandwidth_mhz": "Table I bandwidth.",
    "tx_power_w": "Table I transmit power.",
    "noise_psd_dbm_hz": "Table I thermal-noise PSD.",
    "rician_k_db": "Table I Rician K factor.",
    "atmospheric_attenuation_coefficient_db_per_km": "Table I attenuation coefficient.",
    "slot_duration_s": "Paper slot duration.",
    "episode_duration_s": "Paper episode duration.",
    "episodes": "Paper baseline training horizon.",
    "hidden_layers": "Paper network width/depth.",
    "activation": "Paper activation surface.",
    "optimizer": "Paper optimizer family.",
    "learning_rate": "Paper learning rate.",
    "discount_factor": "Paper discount factor.",
    "batch_size": "Paper batch size.",
    "objective_weights": "Baseline objective weights used for the primary run.",
}

_RECOVERED_FIELDS: dict[str, str] = {
    "configResolved.paper_backed_weight_rows.table_ii": "Recovered Table II weight rows.",
    "configResolved.paper_ranges.user_count": "Recovered user-count figure range.",
    "configResolved.paper_ranges.satellite_count": "Recovered satellite-count figure range.",
    "configResolved.paper_ranges.user_speed_kmh": "Recovered user-speed figure range.",
    "configResolved.paper_ranges.satellite_speed_km_s": "Recovered satellite-speed figure range.",
}


def _build_provenance_entry(
    *,
    primary_classification: str,
    source: str,
    note: str,
    assumption_id: str | None = None,
    source_chain: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "primaryClassification": primary_classification,
        "source": source,
        "note": note,
    }
    if assumption_id is not None:
        payload["assumptionId"] = assumption_id
    if source_chain is not None:
        payload["sourceChain"] = list(source_chain)
    return payload


def build_provenance_map(
    cfg: dict[str, Any],
    metadata: RunMetadataV1,
) -> dict[str, Any]:
    paper = cfg.get("paper", {})
    baseline = cfg.get("baseline", {})
    resolved = cfg.get("resolved_assumptions", {})
    fields: dict[str, Any] = {
        "configResolved.paper.paper_id": _build_provenance_entry(
            primary_classification="paper-backed",
            source="paper-source/ref PDF",
            note="Portable paper identifier for PAP-2024-MORL-MULTIBEAM.",
        ),
        "configResolved.paper.title": _build_provenance_entry(
            primary_classification="paper-backed",
            source="paper-source/ref PDF",
            note="Portable paper title snapshot.",
        ),
        "manifest.coordinateFrame": _build_provenance_entry(
            primary_classification="platform-visualization-only",
            source="Phase 03A replay bundle contract",
            note="Consumer-facing rendering coordinate metadata; not a training authority input.",
        ),
        "timeline.stepTrace.userPosition.localTangentKm": _build_provenance_entry(
            primary_classification="platform-visualization-only",
            source="Phase 03A replay bundle contract",
            note="Derived export convenience for frontend rendering; geodetic user positions remain the underlying truth.",
        ),
        "timeline.stepTrace.rewardVector": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Per-slot replay reward components derived from the exported checkpoint and environment semantics.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
        "timeline.stepTrace.selectedServing": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Serving-beam truth selected by the exported checkpoint during deterministic replay.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
        "timeline.stepTrace.satelliteStates": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Per-slot satellite geometry derived from the replay timeline.",
            source_chain=["reproduction-assumption"],
        ),
        "timeline.stepTrace.beamStates": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Per-slot beam geometry derived from the replay timeline.",
            source_chain=["reproduction-assumption"],
        ),
        "timeline.stepTrace.policyDiagnostics": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the selected checkpoint",
            note="Optional producer-owned greedy policy diagnostics derived from the same masked decision surface as the exported selected serving action.",
            source_chain=["artifact-derived"],
        ),
        "evaluation.summary.training_summary": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="training_log.json and run_metadata.json",
            note="Aggregated training-run summary exported from the completed artifact.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
        "evaluation.summary.best_eval_summary": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="run_metadata.json",
            note="Evaluation summary captured during checkpoint selection.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
    }

    if isinstance(paper, dict):
        if "paper_id" in paper:
            fields["configResolved.paper.paper_id"] = _build_provenance_entry(
                primary_classification="paper-backed",
                source="paper-source/ref PDF",
                note=f"Portable paper identifier: {paper['paper_id']}.",
            )
        if "title" in paper:
            fields["configResolved.paper.title"] = _build_provenance_entry(
                primary_classification="paper-backed",
                source="paper-source/ref PDF",
                note="Portable paper title snapshot.",
            )

    if isinstance(baseline, dict):
        for key, note in _PAPER_BACKED_BASELINE_FIELDS.items():
            if key in baseline:
                fields[f"configResolved.baseline.{key}"] = _build_provenance_entry(
                    primary_classification="paper-backed",
                    source="paper-source/ref PDF or paper-backed baseline envelope",
                    note=note,
                )

    for field_path, note in _RECOVERED_FIELDS.items():
        fields[field_path] = _build_provenance_entry(
            primary_classification="recovered-from-paper",
            source="paper-source/txt_layout and paper-source/catalog",
            note=note,
        )

    if isinstance(resolved, dict):
        for name, block in sorted(resolved.items()):
            if not isinstance(block, dict):
                continue
            fields[
                f"configResolved.resolved_assumptions.{name}"
            ] = _build_provenance_entry(
                primary_classification="reproduction-assumption",
                source="resolved-run config",
                note="Explicit reproduction assumption promoted into the executable config surface.",
                assumption_id=str(block.get("assumption_id", "")) or None,
            )

    fields["manifest.replayTruthMode"] = _build_provenance_entry(
        primary_classification="artifact-derived",
        source="exporter checkpoint selection",
        note="Replay truth source chosen from the exported run artifact.",
        source_chain=["artifact-derived"],
    )
    fields["manifest.optionalPolicyDiagnostics"] = _build_provenance_entry(
        primary_classification="artifact-derived",
        source="Phase 03B additive producer diagnostics export",
        note="Manifest disclosure for the optional policyDiagnostics row surface.",
        source_chain=["artifact-derived"],
    )

    return {
        "bundleSchemaVersion": BUNDLE_SCHEMA_VERSION,
        "classificationLegend": {
            "paper-backed": "Directly supported by the paper or its authority snapshot.",
            "recovered-from-paper": "Recovered from paper tables/figures/text but normalized into executable data.",
            "reproduction-assumption": "Explicit assumption required to make the repo executable and reproducible.",
            "platform-visualization-only": "Consumer/rendering convenience that must not be treated as training authority.",
            "artifact-derived": "Computed from the exported checkpoint and run artifact rather than copied from source authority.",
        },
        "fieldCount": len(fields),
        "producerVersion": str(metadata.package_version or PACKAGE_VERSION),
        "fields": fields,
    }
