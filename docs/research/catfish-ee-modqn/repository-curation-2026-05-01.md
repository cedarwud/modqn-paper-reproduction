# Repository Curation Note

**Date:** `2026-05-01`
**Scope:** tracked-file and workspace-adjacent cleanup for
`modqn-paper-reproduction` after the Catfish / HOBS active-TX EE feasibility
gates.

## Summary

The repo was cleaned and consolidated in commit:

```text
5aa5f00 Consolidate Catfish and HOBS EE feasibility gates
```

That commit removed ignored local training outputs from `artifacts/` and kept
only source, configs, tests, docs, and readable `review.md` evidence. This note
records the follow-up curation decision for already-tracked files and
workspace-adjacent folders.

## Tracked Files

No tracked files were deleted in this pass.

Reason: the tracked historical configs, execution reports, review docs, and
small artifact `review.md` files are not active development targets, but they
are part of the evidence chain that explains why routes are promoted, blocked,
or closed. Removing them would make the current claim boundary harder to audit.

Keep:

1. baseline reproduction artifacts and reviews,
2. Phase `02` / `03` EE-MODQN failure reports,
3. Phase `05` / `07` Catfish reports and bounded configs,
4. RA-EE reports and bounded configs,
5. HOBS active-TX EE / SINR / DPC / Route `D` reports and bounded configs,
6. tests that protect disabled-by-default / namespace-gated behavior.

Do not treat "not the next route" as "safe to delete." A negative result is
still evidence.

## Imported From Workspace

The workspace-level folder:

```text
/home/u24/papers/energy-efficient/
```

was moved into:

```text
docs/research/catfish-ee-modqn/energy-efficient/
```

Reason: these files directly define and review the active-TX EE formula policy
and the MODQN r1-to-HOBS-active-TX-EE design. Keeping them outside the repo made
the next EE-MODQN design gate depend on an untracked workspace folder.

The imported folder is supporting review material. Current implementation
authority remains:

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
```

## Not Imported

The following workspace folders were inspected conceptually but should not be
copied into this repo by default:

| Folder | Decision | Reason |
|---|---|---|
| `paper-catalog/` | do not import | Source-of-truth paper catalog and PDFs are workspace-level literature authority, not repo-local development files. |
| `system-model-refs/` | do not import | Canonical cross-paper formula synthesis should stay central; copy only bounded excerpts if a future report needs them. |
| `catfish/` / `catfish-independent-synthesis/` | do not bulk-import | These folders are now historical / concept-source material for the reopened Multi-Catfish redesign line. They must not override `2026-05-02-multi-catfish-redesign-plan.md`; import only bounded excerpts if a future Catfish-specific gate needs them. |
| `ntn-showcase-stack/` | do not import | Cross-repo coordination / presentation framing only; it must not override this repo's evidence chain. |
| `modqn-ee-presentation-pack/` | do not import | Derived slide / presentation package; useful for communication, not for implementation authority. |
| `modqn-multicatfish-presentation-pack/` | do not import | Derived slide / presentation package; useful for communication, not for implementation authority. |
| `ntn-sim-core/` | do not import | Separate simulator / presentation target. Use via explicit integration contracts only. |

## 2026-05-02 Update: Current Next-Step Boundary

For the EE formula route, the next useful work is not repository cleanup and
not more Route `D` training. The active-TX / CP-base routes have since reached
their stop-loss boundary, while HEA-MODQN became the scoped positive thesis
result. The accepted EE formula surface is now fixed to scoped HEA:

```text
EE_HO = total_bits / (communication_energy_joules + E_HO_total_joules)
E_HO_total_joules = handover_count * E_HO
```

Catfish / Multi-Catfish still must not be imported or attached as an EE repair
mechanism for active-TX, CP-base, Phase `06`, or general communication-only EE.
However, the Catfish concept has been reopened as a separate Multi-Catfish-first
redesign line over the fixed scoped HEA surface. The current authority for that
line is:

```text
docs/research/catfish-ee-modqn/2026-05-02-multi-catfish-redesign-plan.md
```

Old single-first and Phase `05` materials are now historical evidence and
failure constraints, not the next execution plan.
