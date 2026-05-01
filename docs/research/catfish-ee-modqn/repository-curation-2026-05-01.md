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
| `catfish/` / `catfish-independent-synthesis/` | do not import now | Catfish is not the next EE repair path. Keep as external source material unless a future Catfish-specific gate reopens. |
| `ntn-showcase-stack/` | do not import | Cross-repo coordination / presentation framing only; it must not override this repo's evidence chain. |
| `modqn-ee-presentation-pack/` | do not import | Derived slide / presentation package; useful for communication, not for implementation authority. |
| `modqn-multicatfish-presentation-pack/` | do not import | Derived slide / presentation package; useful for communication, not for implementation authority. |
| `ntn-sim-core/` | do not import | Separate simulator / presentation target. Use via explicit integration contracts only. |

## Current Next-Step Boundary

For EE-MODQN continuation, the next useful work is not repository cleanup and
not more Route `D` training. It is a new anti-collapse / capacity / assignment
design gate that addresses:

```text
all_evaluated_steps_one_active_beam = true
```

Catfish / Multi-Catfish should not be imported or attached as the next EE repair
mechanism unless that base anti-collapse gate first passes.
