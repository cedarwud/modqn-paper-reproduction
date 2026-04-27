# MODQN Reproduction Assumption Register

This register tracks implementation choices required for `PAP-2024-MORL-MULTIBEAM` that are not fully fixed by the paper.

| ID | Topic | Current Need | Status | Blocking Phase | Must Close Before Code | Affected Surface | Proposed Direction | Notes |
|---|---|---|---|---|---|---|---|---|
| `ASSUME-MODQN-REP-001` | Orbit shell layout | exact STK shell layout is not disclosed | accepted | Phase 01 | yes | env, export, UI | single circular orbital plane at `780 km` with `4` equally spaced satellites; disclosed Walker-like proxy `δ(4/1/0)` | Controls geometry and visibility truth |
| `ASSUME-MODQN-REP-002` | Beam geometry and beam centers | paper gives `7` beams/satellite but not exact footprint geometry | accepted | Phase 01 | yes | env, export, UI | deterministic `7`-beam hex layout: `1` nadir center plus `6` ring beams spaced by `θ_3dB` | Ring beams are indexed clockwise after the center beam |
| `ASSUME-MODQN-REP-003` | Numeric `φ1` and `φ2` | paper gives only `0 < φ1 < φ2` | accepted | Phase 01 | yes | reward, training, plots | fixed pair `φ1 = 0.5`, `φ2 = 1.0` | Keeps handover cost on the same order as throughput reward; sensitivity runs may vary the ratio |
| `ASSUME-MODQN-REP-004` | Epsilon schedule | paper states `ε-greedy` but not decay law | implemented | Phase 01 | no | training | linear decay: `ε_start=1.0`, `ε_end=0.01`, `decay_episodes=7000` | Standard linear schedule over ~78% of training; remaining episodes run at ε_end |
| `ASSUME-MODQN-REP-005` | Target-network update cadence | target network is mentioned but update period is not given | implemented | Phase 01 | no | training | hard copy every `50` episodes | 50 episodes = 500 environment steps between target syncs |
| `ASSUME-MODQN-REP-006` | Replay capacity | replay memory exists but buffer size is not disclosed | implemented | Phase 01 | no | training | FIFO buffer, capacity `50000` transitions, uniform sampling | Holds ~50 episodes of 100-user data; batch_size=128 samples from this |
| `ASSUME-MODQN-REP-007` | Policy-sharing mode | paper treats each user as an agent but does not fully fix parameter sharing semantics | implemented | Phase 01 | no | training, evaluation | `shared-policy`: one DQN per objective, parameters shared across all users | Each user's transition feeds the same replay buffer and trains the same network |
| `ASSUME-MODQN-REP-008` | Noise semantics | paper gives `-174 dBm/Hz` and bandwidth sharing; implementation must decide exact noise computation path | accepted | Phase 01 | yes | env, reward | `σ² = N₀ × B` in linear units; per-user bandwidth sharing stays in Eq. `(3)` only | Avoids double-counting user sharing in both rate and noise |
| `ASSUME-MODQN-REP-009` | Atmospheric attenuation sign/form | OCR/catalog transfer of the attenuation term is sign-sensitive | accepted | Phase 01 | yes | env, reward, upstream-source-chain | primary run uses the paper-published sign with explicit anomaly disclosure; required sensitivity run uses the corrected lossy sign | Primary expression is paper-faithful even though it yields gain for typical values |
| `ASSUME-MODQN-REP-010` | Evaluation aggregation | paper shows plots but does not fully specify averaging/seed aggregation mechanics | accepted | Phase 01 | no | evaluation, plots, export | `5`-seed mean and standard deviation over the declared evaluation seed set | Primary plots/tables report mean and carry std plus per-seed raw outputs in artifacts |
| `ASSUME-MODQN-REP-011` | Topology handling strategy | state/action dimensionality changes when satellite count changes | accepted | Phase 01 | yes | env, training, export | per-topology retrain | Fixed baseline rule for satellite-count sweep |
| `ASSUME-MODQN-REP-012` | Action eligibility and mask semantics | invisible or out-of-topology beams must be invalid actions | accepted | Phase 01 | yes | env, training, export, UI | action catalog uses stable `satellite-major, beam-minor` ordering; invalid actions receive `Q = -∞` before argmax; frozen baseline uses `satellite-visible-all-beams`, while the explicit Phase `01F` follow-on may opt into `nearest-beam-per-visible-satellite` | Includes candidate ordering semantics, the default horizon-visibility proxy, and the opt-in beam-aware follow-on mask surface |
| `ASSUME-MODQN-REP-013` | State encoding and normalization | paper defines symbolic state, not tensor encoding | implemented | Phase 01 | no | training, export | flat concat `[access, log1p(snr), offsets/100km, loads/U]`; `state_dim = 5 * L * K` | SNR log1p is bounded and monotonic; offset and load normalization are explicit in config |
| `ASSUME-MODQN-REP-014` | Trace source and STK sampling import | paper says STK but not the exact runtime import path | open | Phase 01 | no | env, export | recovered trace or disclosed synthetic substitute | Affects replay and geometry provenance |
| `ASSUME-MODQN-REP-015` | Checkpoint selection rule | paper does not define best-vs-final checkpoint reporting | implemented | Phase 01 | yes | training, plots, export | primary report uses the final-episode policy; secondary disclosure reports the best evaluation weighted reward checkpoint | Final-episode and best-eval checkpoint persistence are implemented; best-eval selection uses mean weighted reward over the configured evaluation seed set at the surfaced trainer cadence and at the final episode |
| `ASSUME-MODQN-REP-016` | Figure discrete point-set protocol | paper ranges are known but exact executable x-axis point sets are not yet frozen | accepted | Phase 01 | yes | evaluation, plots, export | resolved-run config owns discrete points: users `{40,60,80,100,120,140,160,180,200}`, satellites `{2,3,4,5,6,7,8}`, user speed `{30,60,90,120,150}`, satellite speed `{7.0,7.2,7.4,7.6,7.8}` | Prevents hardcoded plot drift |
| `ASSUME-MODQN-REP-017` | Comparator training protocol | `DQN_scalar` and `DQN_throughput` need explicit retrain/eval rules | accepted | Phase 01 | no | training, evaluation | `DQN_scalar` retrain per weight row | Keeps baseline comparisons fair |
| `ASSUME-MODQN-REP-018` | Seed and RNG domain policy | paper does not disclose seed plan | accepted | Phase 01 | yes | training, evaluation, export | fixed split seeds: train `42`, env `1337`, mobility `7`, eval `{100,200,300,400,500}` | Required for reproducibility claims and paired with `ASSUME-MODQN-REP-010` aggregation |
| `ASSUME-MODQN-REP-019` | r3 gap beam scope | paper Eq. defines r3 as max-min beam throughput gap but does not specify whether the gap spans only occupied beams or all reachable beams | implemented | Phase 01 | no | env, reward | gap spans all **reachable** beams (union of all users' visibility masks); empty reachable beams contribute throughput `0` to the min | Previous occupied-only semantics yielded `gap=0` when all users concentrated on one beam, hiding the worst load imbalance |
| `ASSUME-MODQN-REP-020` | User heading stride | paper does not specify a user mobility heading model | implemented | Phase 01 | no | env | deterministic per-user heading via irrational stride `heading = (uid × 2.3998277) mod 2π` | Provides quasi-uniform angular spread without an RNG draw; constant surfaced as `USER_HEADING_STRIDE_RAD` |
| `ASSUME-MODQN-REP-021` | User scatter radius | paper does not specify user spatial distribution | implemented | Phase 01 | no | env | users uniformly scattered in a `50 km` radius circle around the configured ground point | Constant surfaced as `USER_SCATTER_RADIUS_KM`; reasonable urban/suburban coverage assumption |
| `ASSUME-MODQN-REP-022` | User area geometry | paper gives the user-area center and rectangular extent, but the executable runtime must still define a concrete within-area sampling rule | accepted | Phase 01B | no | env, export | paper-faithful center point with `uniform-rectangle` sampling inside `200 km × 90 km` | Closes the highest-impact scenario mismatch versus the frozen baseline's circular scatter proxy |
| `ASSUME-MODQN-REP-023` | Random wandering mobility rule | paper states `random wandering` but does not disclose the exact per-slot heading-update law | accepted | Phase 01B | no | env, export | maintain a per-user heading and apply bounded uniform turn noise each slot | Keeps the mobility family paper-faithful while surfacing the unresolved turn-law details |

## Resolution Rule

Each assumption should move through:

1. `open`
2. `proposed`
3. `accepted`
4. `implemented`

These lifecycle states are separate from the provenance taxonomy defined in Phase 01 SDD:

1. `paper-backed`
2. `recovered-from-paper`
3. `reproduction-assumption`
4. `platform-visualization-only`

## Disclosure Rule

No run artifact may hide these assumptions. Every completed run must record:

1. which assumption IDs were active
2. the chosen values
3. why the value was chosen
