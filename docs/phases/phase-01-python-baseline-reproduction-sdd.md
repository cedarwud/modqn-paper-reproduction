# Phase 01: Python Baseline Reproduction SDD

**Status:** Active SDD  
**Project:** `modqn-paper-reproduction`  
**Primary goal:** produce a paper-consistent baseline reproduction for `PAP-2024-MORL-MULTIBEAM` before any `ntn-sim-core` integration

## 1. Scope

### 1.1 In Scope

1. standalone Python environment for the baseline paper
2. paper-faithful `state / action / reward` implementation
3. `MODQN` training path
4. comparator baselines:
   - `RSS_max`
   - `DQN_throughput`
   - `DQN_scalar`
5. evaluation sweeps and figure-ready outputs for:
   - `Table II`
   - `Fig. 3`
   - `Fig. 4`
   - `Fig. 5`
   - `Fig. 6`
6. explicit assumption tracking

### 1.2 Out of Scope

1. `ntn-sim-core` runtime integration
2. 3D replay/visualization
3. generalized 3GPP-accurate path-loss replacement of the paper model
4. broader multi-paper platform abstractions

### 1.3 Provenance Taxonomy

All Phase 01 surfaces must use the same provenance taxonomy:

1. `paper-backed`
   Directly stated in the paper or source PDF chain.
2. `recovered-from-paper`
   Recovered from the paper source chain but not already normalized into a machine-ready field.
3. `reproduction-assumption`
   Concrete implementation choice required because the paper does not fully fix the value or rule.
4. `platform-visualization-only`
   Consumer-only augmentation used by later UI or replay layers and not part of the Phase 01 trainer truth.

The assumption register lifecycle states (`open`, `proposed`, `accepted`, `implemented`) are separate from this provenance taxonomy.

## 2. Authority Sources

Use these sources in order:

1. paper PDF and regenerated OCR under `../paper-catalog/`
2. `../paper-catalog/catalog/PAP-2024-MORL-MULTIBEAM.json`
3. `docs/assumptions/modqn-reproduction-assumption-register.md`

`../system-model-refs/` is allowed only as a cross-check to identify where the paper is simplified, not as a silent override of the reproduction model.

## 3. Paper Baseline Summary

### 3.1 Scenario

The paper baseline fixes:

1. `4` satellites
2. `7` beams per satellite
3. `100` users
4. altitude `780 km`
5. user speed `30 km/h`
6. satellite speed `7.4 km/s`
7. carrier `20 GHz`
8. bandwidth `500 MHz`
9. per-link transmit power `2 W`
10. noise PSD `-174 dBm/Hz`
11. Rician `K = 20 dB`
12. atmospheric attenuation coefficient envelope `0.05 dB/km`

The atmospheric term now follows a disclosed `reproduction-assumption`:

1. the primary baseline run uses the paper-published sign as written
2. run metadata must flag that the published sign yields gain rather than loss for typical values
3. a sensitivity run must also evaluate the corrected lossy sign `A(d) = 10^{-3dχ/(10h)}`

### 3.2 State

Per user:

```text
s_i(t) = (u_i(t), G_i(t), Γ(t), N(t))
```

where:

1. `u_i(t)` is the current access vector
2. `G_i(t)` is the channel quality to all beams
3. `Γ(t)` is the beam-location surface
4. `N(t)` is the number of users served by each beam

Pending final closure of `ASSUME-MODQN-REP-013`, the default Phase 01 tensor layout is:

1. all beam-indexed terms are flattened in stable `satellite-major, beam-minor` order
2. `Γ(t)` is encoded as per-beam user-relative local-tangent offsets `(east_km, north_km)`
3. `N(t)` records pre-decision beam loads from the current slot state, not projected post-action loads
4. any normalization layer must be disclosed explicitly under `ASSUME-MODQN-REP-013` and must not silently replace the raw exported values

### 3.3 Action

One-hot beam selection across all available beams.

### 3.3A Topology And Dimensionality Strategy

Phase 1 fixes the topology strategy as follows:

1. within one training run, the policy network input/output dimensions are fixed to that run's topology
2. action catalog ordering is stable and uses `satellite-major, beam-minor` indexing
3. invalid actions are masked at decision time if a beam is outside:
   - the current run topology envelope
   - the current visibility/eligibility set
4. satellite-count sweep points that change `L` are treated as distinct topology families
5. the default Phase 1 rule for those topology families is `per-topology retrain`, not one shared fixed-max output network

This is a `reproduction-assumption` rather than a `paper-backed` fact:

1. `ASSUME-MODQN-REP-011` locks `per-topology retrain` as the current baseline rule
2. `ASSUME-MODQN-REP-012` locks action-mask semantics and candidate ordering
3. the default visibility proxy treats a satellite as eligible only when its line-of-sight elevation is above `0°`

### 3.4 Reward

Three-objective reward vector:

1. `r1`: throughput
2. `r2`: handover penalty
3. `r3`: negative max-min beam throughput gap divided by user count

### 3.5 Training Protocol

The baseline protocol is:

1. 3 parallel DQNs
2. hidden layers `[100, 50, 50]`
3. `tanh`
4. `Adam`
5. learning rate `0.01`
6. discount factor `0.9`
7. batch size `128`
8. slot `1 s`
9. episode length `10 s`
10. episodes `9000`
11. objective weights `[0.5, 0.3, 0.2]` as the primary baseline

### 3.6 Figure And Weight Protocol

Phase 1 distinguishes between:

1. `paper-backed` ranges
2. `recovered-from-paper` weight rows
3. `reproduction-assumption` executable point sets

Table II weight rows currently treated as `recovered-from-paper` rows are:

1. `[1.0, 1.0, 1.0]`
2. `[1.0, 0.0, 0.0]`
3. `[0.0, 1.0, 0.0]`
4. `[0.0, 0.0, 1.0]`
5. `[0.5, 0.3, 0.2]`
6. `[0.5, 0.2, 0.3]`
7. `[0.3, 0.5, 0.2]`
8. `[0.2, 0.5, 0.3]`
9. `[0.4, 0.4, 0.2]`
10. `[0.4, 0.2, 0.4]`
11. `[0.2, 0.4, 0.4]`

The execution protocol for weight evaluation is:

1. `MODQN` is trained once per topology family and evaluated across weight rows using scalarization at action selection
2. `DQN_scalar` is retrained per weight row because its reward is weight-specific
3. `DQN_throughput` and `RSS_max` do not use weight-row retraining

Paper-backed sweep ranges are:

1. users: `40` to `200`
2. satellites: `2` to `8`
3. user speed: `30` to `150 km/h`
4. satellite speed: `7.0` to `7.8 km/s`

The exact discrete point sets used to generate plots remain `reproduction-assumption` values until explicitly frozen in the resolved-run config under `ASSUME-MODQN-REP-016`.

### 3.7 Configuration Surfaces

Phase 1 uses two config roles:

1. paper-envelope config
2. resolved-run config

The paper-envelope config records:

1. `paper-backed` parameters
2. `recovered-from-paper` weights
3. `paper-backed` ranges
4. `reproduction-assumption` references

The resolved-run config records:

1. concrete assumption values
2. concrete sweep point sets
3. seeds
4. checkpoint rule
5. aggregation rule
6. executable trainer settings

No training run may start from the paper-envelope config alone.

## 4. Implementation Modules

Phase 1 should land these modules:

1. `src/modqn_paper_reproduction/contracts.py`
   Shared dataclasses and run-result contracts.
2. `src/modqn_paper_reproduction/settings.py`
   Paper-backed parameters plus explicit assumption slots.
3. `src/modqn_paper_reproduction/env/`
   Environment state transition and reward logic.
4. `src/modqn_paper_reproduction/algorithms/`
   `MODQN` and DQN variants.
5. `src/modqn_paper_reproduction/baselines/`
   `RSS_max`.
6. `src/modqn_paper_reproduction/export/`
   Figure-ready tables and sweep data.

## 5. Assumption Policy

Any implementation choice not uniquely fixed by the paper must:

1. receive an `ASSUME-MODQN-REP-*` ID
2. be added to the assumption register
3. be surfaced in run metadata

No hidden constants are allowed for:

1. `φ1` / `φ2`
2. epsilon schedule
3. target update cadence
4. replay capacity
5. orbit layout proxy
6. beam geometry proxy
7. topology handling strategy
8. action masking / eligibility
9. seed and aggregation rules
10. checkpoint selection rule
11. figure point-set selection

### 5.1 Blocking Assumptions For Code Start

The following assumptions must close before the first real Phase 1 training implementation is allowed to claim executable baseline status:

1. `ASSUME-MODQN-REP-001` orbit shell layout / trace source
2. `ASSUME-MODQN-REP-002` beam geometry and beam centers
3. `ASSUME-MODQN-REP-003` numeric `φ1` and `φ2`
4. `ASSUME-MODQN-REP-008` noise semantics
5. `ASSUME-MODQN-REP-009` atmospheric attenuation sign/form
6. `ASSUME-MODQN-REP-011` topology handling strategy
7. `ASSUME-MODQN-REP-012` action masking / eligibility semantics
8. `ASSUME-MODQN-REP-015` checkpoint selection rule
9. `ASSUME-MODQN-REP-016` figure discrete point-set protocol
10. `ASSUME-MODQN-REP-018` seed and aggregation protocol

### 5.2 Reproducibility Rules

Every executable resolved-run config must declare:

1. `train_seed`
2. `environment_seed`
3. `mobility_seed`
4. `evaluation_seed_set`
5. `checkpoint_selection_rule`
6. `metric_aggregation_rule`

If any of these are missing, the run is not a valid reproduction artifact.

## 6. Evaluation Outputs

Phase 1 must produce both raw data and plots.

### 6.1 Required Tables

1. `Table II` weight comparison

### 6.2 Required Figures

1. `Fig. 3`: user-count sweep
2. `Fig. 4`: satellite-count sweep
3. `Fig. 5`: user-speed sweep
4. `Fig. 6`: satellite-speed sweep

Each figure family should emit:

1. objective-detail data
2. weighted-reward data
3. `MODQN` and comparator outputs

### 6.3 Acceptance Shape

Each produced table/figure must be accompanied by:

1. machine-readable CSV or JSON
2. resolved-run manifest with seeds and assumption values
3. explicit comparator list
4. checkpoint rule used for the plotted results
5. aggregation rule used for the plotted results

Plot image generation alone is not sufficient for acceptance.

## 7. Completion Boundary

Phase 1 is complete only when:

1. `MODQN` can train end-to-end
2. all comparator baselines can run
3. one resolved-run config can execute the default paper scenario
4. each required sweep can emit machine-readable data
5. each required table/figure can emit both a plot image and a machine-readable companion artifact
6. every non-`paper-backed` execution choice is labeled with the canonical provenance taxonomy, with missing-paper implementation choices recorded as `reproduction-assumption`
7. the emitted run manifest records seeds, checkpoint rule, and aggregation rule
8. no Phase 1 internal dataclass or CSV header is treated as a frozen external contract

## 8. Claim Boundary

Phase 1 does **not** claim strict exact replication unless all missing paper details are closed.

Default wording must remain:

1. `paper-consistent reproduction`
2. `trend-faithful` where numeric parity is not yet established
3. `reproduction-assumption` for implementation choices required by missing paper detail
