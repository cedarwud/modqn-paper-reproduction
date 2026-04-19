"""Typed bundle-layer models introduced by Phase 04C."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any


def _copy_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(payload)


@dataclass(frozen=True)
class ReplaySampleSubset:
    max_users: int
    max_slots: int | None
    user_indices: tuple[int, ...]
    slot_indices: tuple[int, ...]
    source_full_row_count: int
    source_full_slot_count: int
    source_full_handover_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxUsers": int(self.max_users),
            "maxSlots": None if self.max_slots is None else int(self.max_slots),
            "userIndices": [int(value) for value in self.user_indices],
            "slotIndices": [int(value) for value in self.slot_indices],
            "sourceFullRowCount": int(self.source_full_row_count),
            "sourceFullSlotCount": int(self.source_full_slot_count),
            "sourceFullHandoverEventCount": int(
                self.source_full_handover_event_count
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplaySampleSubset:
        return cls(
            max_users=int(payload["maxUsers"]),
            max_slots=(
                None if payload.get("maxSlots") is None else int(payload["maxSlots"])
            ),
            user_indices=tuple(int(value) for value in payload.get("userIndices", [])),
            slot_indices=tuple(int(value) for value in payload.get("slotIndices", [])),
            source_full_row_count=int(payload["sourceFullRowCount"]),
            source_full_slot_count=int(payload["sourceFullSlotCount"]),
            source_full_handover_event_count=int(
                payload["sourceFullHandoverEventCount"]
            ),
        )


@dataclass(frozen=True)
class ReplaySummary:
    checkpoint_path: str
    checkpoint_kind: str
    policy_episode: int
    timeline_seed: int
    replay_seed_source: str
    slot_index_offset: int
    row_count: int
    slot_count: int
    handover_event_count: int
    reward_weights: tuple[float, ...]
    diagnostics: dict[str, Any]
    sample_subset: ReplaySampleSubset | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "checkpointPath": self.checkpoint_path,
            "checkpointKind": self.checkpoint_kind,
            "policyEpisode": int(self.policy_episode),
            "timelineSeed": int(self.timeline_seed),
            "replaySeedSource": self.replay_seed_source,
            "slotIndexOffset": int(self.slot_index_offset),
            "rowCount": int(self.row_count),
            "slotCount": int(self.slot_count),
            "handoverEventCount": int(self.handover_event_count),
            "rewardWeights": [float(value) for value in self.reward_weights],
            "diagnostics": _copy_mapping(self.diagnostics),
        }
        if self.sample_subset is not None:
            payload["sampleSubset"] = self.sample_subset.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplaySummary:
        sample_subset = payload.get("sampleSubset")
        return cls(
            checkpoint_path=str(payload["checkpointPath"]),
            checkpoint_kind=str(payload["checkpointKind"]),
            policy_episode=int(payload["policyEpisode"]),
            timeline_seed=int(payload["timelineSeed"]),
            replay_seed_source=str(payload["replaySeedSource"]),
            slot_index_offset=int(payload.get("slotIndexOffset", 1)),
            row_count=int(payload["rowCount"]),
            slot_count=int(payload["slotCount"]),
            handover_event_count=int(payload["handoverEventCount"]),
            reward_weights=tuple(
                float(value) for value in payload.get("rewardWeights", [])
            ),
            diagnostics=_copy_mapping(payload.get("diagnostics", {})),
            sample_subset=(
                None
                if not isinstance(sample_subset, dict)
                else ReplaySampleSubset.from_dict(sample_subset)
            ),
        )

    def with_checkpoint_path(self, checkpoint_path: str) -> ReplaySummary:
        return replace(self, checkpoint_path=str(checkpoint_path))

    def with_trimmed_subset(
        self,
        *,
        row_count: int,
        slot_count: int,
        handover_event_count: int,
        max_users: int,
        max_slots: int | None,
        user_indices: list[int],
        slot_indices: list[int],
        source_full_row_count: int,
        source_full_slot_count: int,
        source_full_handover_event_count: int,
    ) -> ReplaySummary:
        return replace(
            self,
            row_count=int(row_count),
            slot_count=int(slot_count),
            handover_event_count=int(handover_event_count),
            sample_subset=ReplaySampleSubset(
                max_users=int(max_users),
                max_slots=None if max_slots is None else int(max_slots),
                user_indices=tuple(int(value) for value in user_indices),
                slot_indices=tuple(int(value) for value in slot_indices),
                source_full_row_count=int(source_full_row_count),
                source_full_slot_count=int(source_full_slot_count),
                source_full_handover_event_count=int(
                    source_full_handover_event_count
                ),
            ),
        )
