# Agent File Size Audit - 2026-05-02

## Scope

This audit reviews `modqn-paper-reproduction` for files that are large
enough to make agent reading, review, or targeted editing unnecessarily
expensive. No source files were split or refactored in this pass.

The audit intentionally separates:

1. tracked text files, because these are the most likely files an agent will
   read during code work
2. tracked binary/reference files, because they may be large but are usually
   not opened as text
3. ignored local artifacts, because they can still affect broad filesystem
   scans even though they are not part of the committed authority surface

## Method

Commands used for the inspection:

```bash
git grep -Il '' | xargs wc -l | sort -nr | head -100
git grep -Il '' | xargs du -b | sort -nr | head -100
git ls-files -z | xargs -0 du -b | sort -nr | head -80
find . -path ./.git -prune -o -path ./.venv -prune -o -path ./.pytest_cache -prune -o -path '*/__pycache__' -prune -o -type f -size +1M -printf '%p\n' | wc -l
find artifacts -path '*/checkpoints/*.pt' -type f -printf '%s\n' | awk '{n++; s+=$1} END {printf "count=%d total_bytes=%d total_mib=%.2f\n", n, s, s/1048576}'
```

Additional AST-based inspection measured top-level function/class count and
largest function/method span for the largest Python files.

## Size Summary

- Tracked files: 433
- Tracked text files: 428
- Tracked bytes: 11.79 MiB
- Tracked text bytes: 4.02 MiB
- Tracked text lines: 101,184
- Tracked text files with at least 2,000 lines: 4
- Tracked text files with at least 1,000 lines: 19
- Tracked text files with at least 80 KB: 5
- Non-cache, non-venv files larger than 1 MiB in the local working tree: 81
- Ignored checkpoint files under `artifacts/**/checkpoints/*.pt`: 80 files,
  87.89 MiB total

## Thresholds Used

These thresholds are agent-readability heuristics, not hard repository rules:

- High concern: text file above 2,000 lines, above 80 KB, or containing a
  single function/method above roughly 250 lines.
- Medium concern: text file above 1,000 lines or above 40 KB, especially when
  it mixes several responsibilities.
- Low concern: large binary/reference artifact, generated fixture, or
  cohesive implementation file that is rarely edited as a whole.

## High Concern Split Candidates

### `src/modqn_paper_reproduction/config_loader.py`

- Lines: 2,312
- Size: 98,192 bytes
- Top-level definitions: 19
- Largest function: `build_trainer_config`, 1,840 lines
- Assessment: this is the highest-value split candidate for agent
  readability. Most of the file's cost is concentrated in one function that
  parses defaults, experiment metadata, Catfish settings, HOBS/EE surfaces,
  reward settings, and trainer-construction fields.
- Recommended direction if splitting is authorized:
  - keep public entry points stable in `config_loader.py`
  - extract trainer-config parsing into focused helpers grouped by concern
  - move Catfish/Phase 05/Phase 07/HOBS option parsing into narrow internal
    parser functions or modules
  - keep `build_trainer_config` as a small orchestration layer that returns
    `TrainerConfig`
- Risk: high regression risk because config parsing is a central contract.
  Any split should be test-backed and mostly mechanical.

### `src/modqn_paper_reproduction/runtime/trainer_spec.py`

- Lines: 1,116
- Size: 53,748 bytes
- Classes: 3
- Largest method: `TrainerConfig.__post_init__`, 918 lines
- Assessment: the dataclass itself is understandable, but validation is
  concentrated in one very large post-init method. This is expensive for an
  agent to inspect and risky to edit surgically.
- Recommended direction if splitting is authorized:
  - keep `TrainerConfig` as the public data model
  - extract validation/coercion blocks into private validator helpers
  - consider a `runtime/trainer_config_validation.py` helper module if local
    imports stay clean
- Risk: medium-high because validation ordering can encode compatibility
  behavior. A split should preserve error messages and field semantics.

### `src/modqn_paper_reproduction/env/step.py`

- Lines: 2,268
- Size: 89,056 bytes
- Classes: 8
- Largest method: `StepEnvironment._build_states_and_masks`, 277 lines
- Assessment: large because it owns typed config, result dataclasses,
  environment stepping, reward computation, DPC/power helpers, mobility, and
  diagnostics. It is navigable due to section comments, but still too broad
  for repeated agent edits.
- Recommended direction if splitting is authorized:
  - keep `StepEnvironment` import compatibility stable
  - extract pure helper areas first: power profiles, mobility helpers, reward
    helpers, and result/config dataclasses
  - avoid changing physics/reward behavior during the split
- Risk: high because this is a core simulator surface. Prefer small,
  behavior-preserving moves with existing `tests/test_step.py` coverage.

### `src/modqn_paper_reproduction/analysis/ra_ee_06b_association_proposal_refinement.py`

- Lines: 2,761
- Size: 113,803 bytes
- Top-level definitions: 60
- Largest function: `export_ra_ee_06b_association_proposal_refinement`,
  272 lines
- Assessment: a full experimental pipeline is packed into one module:
  settings, policy selection, active-set refinement, rollout, per-step rows,
  summaries, guardrails, decision logic, and report writing.
- Recommended direction if this analysis line remains active:
  - extract shared settings/config parsing
  - extract association-policy and active-set refinement logic
  - extract summary/guardrail/review writers
  - keep the export function as the orchestration boundary
- Risk: medium. It is easier to split than core runtime files, but analysis
  output compatibility must be preserved.

### `src/modqn_paper_reproduction/analysis/ra_ee_07_constrained_power_allocator_distillation.py`

- Lines: 2,727
- Size: 110,180 bytes
- Top-level definitions: 55
- Largest function: `export_ra_ee_07_constrained_power_allocator_distillation`,
  293 lines
- Assessment: similar to RA-EE-06B, this combines settings, allocator search,
  fixed-snapshot evaluation, association diagnostics, guardrail decisions,
  and report writing.
- Recommended direction if this analysis line remains active:
  - extract allocator implementations from export/report code
  - extract summary and guardrail helpers
  - consider sharing common RA-EE pipeline utilities with RA-EE-05/06/06B/09
- Risk: medium. Preserve generated CSV/JSON/review shapes.

## Medium Concern Files

These files are large enough to slow agent reading, but they are less urgent
than the high-concern files or are relatively cohesive.

| File | Lines | Bytes | Assessment |
| --- | ---: | ---: | --- |
| `src/modqn_paper_reproduction/analysis/ra_ee_06_association_counterfactual_oracle.py` | 1,740 | 69,449 | Candidate for the same pipeline split pattern as RA-EE-06B. |
| `src/modqn_paper_reproduction/analysis/ra_ee_02_oracle_power_allocation.py` | 1,495 | 60,187 | Split only if the oracle/power audit line remains active. |
| `src/modqn_paper_reproduction/analysis/_ra_ee_09_compare.py` | 1,485 | 62,509 | Large comparison/report pipeline; could split aggregate, decision, and review output helpers. |
| `src/modqn_paper_reproduction/analysis/ra_ee_05_fixed_association_robustness.py` | 1,458 | 57,182 | Natural shared-utility donor for RA-EE-06/06B/07 if refactoring is authorized. |
| `src/modqn_paper_reproduction/algorithms/catfish_modqn.py` | 1,224 | 51,816 | Cohesive trainer class, but `train`, replay routing, and diagnostics could be separated later. |
| `src/modqn_paper_reproduction/algorithms/modqn.py` | 1,164 | 44,709 | Cohesive baseline trainer; split only when touching training loop internals. |
| `src/modqn_paper_reproduction/analysis/_ra_ee_09_replay.py` | 1,144 | 48,897 | Medium split candidate around control/candidate replay and review generation. |
| `src/modqn_paper_reproduction/analysis/ra_ee_08_metrics.py` | 1,076 | 47,493 | Moderate; biggest helper is 223 lines. |
| `src/modqn_paper_reproduction/analysis/ra_ee_04_bounded_power_allocator.py` | 1,024 | 39,102 | Moderate; leave unless active edits resume. |
| `src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_qos_sticky_robustness.py` | 1,024 | 40,278 | Moderate; pipeline/report module. |
| `src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_anti_collapse.py` | 1,013 | 43,815 | Moderate; has a 283-line learned-policy evaluator. |
| `src/modqn_paper_reproduction/algorithms/multi_catfish_modqn.py` | 993 | 40,509 | Just below the line-count threshold, but still a large trainer class. |

## Documentation Readability Candidates

These are not code-splitting priorities, but they matter for agent handoff and
authority scanning.

| File | Lines | Notes |
| --- | ---: | --- |
| `docs/ee-report.md` | 1,303 | Long current report with many sections. If it remains active authority, consider a concise index/current-state front matter and move historical detail into appendices. |
| `docs/research/catfish-ee-modqn/execution-handoff.md` | 1,222 | Important handoff surface; a shorter "current decisions only" companion would reduce agent load. |
| `docs/research/catfish-ee-modqn/00-validation-master-plan.md` | 662 | Large but structured. No immediate split required. |
| `docs/research/catfish-ee-modqn/energy-efficient/claude.md` | 472 | No markdown headings found; harder to skim than its size suggests. Add headings before splitting. |
| `docs/research/catfish-ee-modqn/energy-efficient/codex.md` | 388 | No markdown headings found; add headings if this remains a reference surface. |

## Large Binary, Reference, And Fixture Files

### Tracked reference PDF

- `paper-source/ref/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.pdf`
- Size: 8,087,998 bytes
- Assessment: large but valid authority input. Do not split. Agents should
  prefer the extracted text/layout files unless the original PDF is needed for
  source verification.

### Tracked fixture

- `tests/fixtures/sample-bundle-v1/timeline/step-trace.jsonl`
- Size: 118,792 bytes
- Assessment: largest tracked text fixture by bytes, but acceptable as a
  fixture. Do not split unless tests only need a smaller representative trace.

### Ignored local checkpoints

- `artifacts/**/checkpoints/*.pt`
- Count: 80 files
- Total size: 87.89 MiB
- Largest individual checkpoint: about 1.10 MiB
- Assessment: not a code readability problem and already excluded by
  `.gitignore`, but broad `find`/filesystem scans will encounter them. Do not
  split. Consider cleanup only if disk/runtime hygiene becomes an issue.

## Overall Recommendation

Do not start a broad split immediately. If the goal is to improve future Codex
agent work, the most meaningful first pass would be a small, behavior-preserving
refactor of the central contracts:

1. `config_loader.py`, focused on reducing `build_trainer_config`
2. `runtime/trainer_spec.py`, focused on extracting `TrainerConfig`
   validation blocks
3. `env/step.py`, focused on moving pure helpers and dataclasses while keeping
   `StepEnvironment` stable

The RA-EE analysis modules should be split only if that research line remains
active. They are large, but their risk is mostly review/output compatibility
rather than runtime contract compatibility.

No direct split or source-code refactor was performed by this audit.
