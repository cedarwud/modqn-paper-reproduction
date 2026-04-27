# Read-Only Audit: Baseline Contract vs Reopen Candidates

Date: `2026-04-22`

Scope:

- read-only review of current `modqn-paper-reproduction` baseline semantics
- no trainer/config/code changes
- goal: decide whether more episodes are likely to change the current result surface

Sources reviewed:

- `paper-source/txt_layout/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.layout.txt`
- `docs/assumptions/modqn-reproduction-assumption-register.md`
- `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
- `src/modqn_paper_reproduction/env/step.py`
- `src/modqn_paper_reproduction/env/beam.py`
- `src/modqn_paper_reproduction/env/channel.py`
- `src/modqn_paper_reproduction/baselines/rss_max.py`
- `src/modqn_paper_reproduction/algorithms/modqn.py`
- `artifacts/run-9000/anomaly-review.md`
- one-step rollout trace using:
  - config `configs/modqn-paper-baseline.resolved-template.yaml`
  - checkpoint `artifacts/codex-pilot-200ep-2026-04-22/checkpoints/best-weighted-reward-on-eval.pt`
  - evaluation seed `100`

## Bottom Line

This review did **not** find a simple implementation bug such as:

1. broken epsilon decay
2. target sync not firing
3. sign inversion in the handover penalty
4. an obviously incorrect checkpoint load path

The stronger explanation is structural:

1. current baseline action semantics are close to **satellite-granular**, not richly beam-discriminative
2. current baseline channel semantics preserve the disclosed atmospheric-sign anomaly and therefore keep throughput dominant
3. the fixed `RSS_max` comparator loses much of its intended meaning under those semantics
4. `r3` is globally defined but copied into each user transition, which weakens credit assignment in shared-policy replay

Taken together, these points make "train longer" a weak next move.

## Paper vs Code vs Assumption Mapping

### 1. Beam-level action space exists in the paper, but beam-level channel discrimination is not fixed by the paper

Paper evidence:

1. the paper defines the action/access vector over all beams `u_i(t)` and the per-beam index `κ = (l, v)` in Eq. `(5)` context
2. the paper defines `G_i(t)` as channel quality between user `i` and **all beams**

Relevant lines:

- `paper-source/...layout.txt:90-118`
- `paper-source/...layout.txt:153-169`

Current baseline implementation:

1. the action catalog is beam-level and stable `satellite-major, beam-minor`
2. the validity mask is satellite-level: if satellite `l` is visible, all `V=7` beams under that satellite are marked valid
3. channel is also satellite-level: one `compute_channel()` call per visible satellite, then the same `snr_linear` is copied across that satellite's whole beam block
4. off-axis beam gain is explicitly not modeled

Relevant code:

- `src/modqn_paper_reproduction/env/step.py:548-578`
- `src/modqn_paper_reproduction/env/beam.py:11-16`

Classification:

- `baseline contract`: yes
- authority path:
  - `ASSUME-MODQN-REP-012` locks satellite-visible eligibility and candidate ordering
  - `ASSUME-MODQN-REP-002` locks beam geometry only, not beam-specific channel gain

Interpretation:

This is an accepted baseline contract, not a hidden bug. But it is also not a paper-backed guarantee that beam-level channel quality should collapse to one value per visible satellite. If the scientific question requires meaningful beam-vs-beam channel separation, this is a legitimate reopen candidate.

### 2. Atmospheric sign anomaly is preserved by design, and it keeps throughput dominant

Paper evidence:

1. the layout text preserves the atmospheric factor as
   `A(d_i,l,v(t)) = 10^(+3 d_i,l,v(t) χ / (10 h_l,v(t)))`
2. that expression is sign-sensitive and, as written, yields gain rather than loss for typical positive values

Relevant lines:

- `paper-source/...layout.txt:115-140`

Current baseline implementation:

1. the default `ChannelConfig` uses `AtmosphericSignMode.PAPER_PUBLISHED`
2. `atmospheric_factor()` applies the positive exponent unless the explicit corrected-lossy sensitivity mode is chosen
3. the runtime diagnostics already disclose that `|r1| / |r2| ≈ 982x`

Relevant code:

- `src/modqn_paper_reproduction/env/channel.py:46-58`
- `src/modqn_paper_reproduction/env/channel.py:87-89`
- `src/modqn_paper_reproduction/env/channel.py:200-233`

Relevant assumption:

- `docs/assumptions/modqn-reproduction-assumption-register.md:15`

Classification:

- `baseline contract`: yes
- `reopen candidate`: also yes, if the goal changes from "paper-faithful disclosed baseline" to "physically lossy sensitivity or better reward geometry"

Interpretation:

This is not something more episodes can repair. The baseline intentionally preserves the paper-published sign and explicitly discloses the anomaly. Long training can only optimize inside that reward geometry; it cannot change the geometry itself.

### 3. RSS_max loses explanatory power under the current channel semantics

Paper evidence:

1. the paper describes `RSS_max` as selecting the beam with the best channel quality

Relevant lines:

- `paper-source/...layout.txt:310-319`

Current baseline implementation:

1. `RSS_max` does `argmax(channel_quality)` over valid actions
2. if every beam under one visible satellite shares the same `channel_quality`, `RSS_max` degenerates to the lowest-index valid beam in that block

Relevant code:

- `src/modqn_paper_reproduction/baselines/rss_max.py:32-42`
- `src/modqn_paper_reproduction/env/step.py:563-578`

Classification:

- `baseline contract`: partially
- `reopen candidate`: yes, for comparator interpretability

Interpretation:

Keeping the current semantics is still internally consistent, but the comparator is no longer a strong "best channel beam" baseline in the ordinary sense. If the baseline family is kept as-is, this limitation should be disclosed more explicitly in any interpretation note.

### 4. `r3` is a global signal copied into every user transition

Paper evidence:

1. the paper defines `r3` from the network-wide max-min beam throughput gap
2. the paper also says each agent stores objective rewards in replay memory individually

Relevant lines:

- `paper-source/...layout.txt:188-236`

Current baseline implementation:

1. `_compute_rewards()` computes one global `gap`
2. the same `r3 = -gap / U` is then assigned to every user reward
3. training pushes each user's transition separately into the shared replay buffer, carrying that same global `r3`

Relevant code:

- `src/modqn_paper_reproduction/env/step.py:651-679`
- `src/modqn_paper_reproduction/algorithms/modqn.py:718-734`

Relevant assumption / protocol:

- `docs/assumptions/modqn-reproduction-assumption-register.md:13`
- `docs/assumptions/modqn-reproduction-assumption-register.md:25`

Classification:

- `baseline contract`: yes
- `reopen candidate`: yes, for credit-assignment quality

Interpretation:

This is not obviously incorrect relative to the paper's global `r3` definition, but it does mean the shared-policy replay buffer receives a weakly localized fairness signal. That makes "sticky, low-handover, not-clearly-better-service" behavior plausible without requiring a training bug.

## One-Step Rollout Trace

Checkpoint used:

- `artifacts/codex-pilot-200ep-2026-04-22/checkpoints/best-weighted-reward-on-eval.pt`

Trace setup:

1. resolved-run baseline config
2. evaluation seed `100`
3. user `uid=0`
4. first post-reset decision

Observed facts:

1. valid actions were exactly one 7-beam block: `[0, 1, 2, 3, 4, 5, 6]`
2. all valid actions had the **same** `snr_linear`: `2.314099e-06`
3. beam offsets differed across those 7 actions, so geometry is not collapsed, but channel quality is
4. pre-decision beam loads differed (`5, 18, 12, 20, 15, 14, 16`), which means load is one of the few discriminators left inside the block
5. `RSS_max` selected action `0` because all valid SNRs were tied
6. MODQN selected action `1`
7. MODQN's scalarized-Q margin over the runner-up was only `0.8508`, with top valid scalarized-Q values clustered in `107.6` to `111.1`

Trace excerpt:

```json
{
  "valid_actions": [0, 1, 2, 3, 4, 5, 6],
  "unique_valid_snr_linear": [2.314099e-06],
  "selected_action": 1,
  "rss_max_action": 0,
  "scalarizedMarginToRunnerUp": 0.8508377075195312,
  "reward": {
    "r1": 17.765361785888672,
    "r2": -0.5,
    "r3": -16.853792934417726
  }
}
```

Interpretation:

This trace makes the near-tie mechanism concrete:

1. the decision is still formally beam-level
2. but within a visible satellite, channel quality is fully tied
3. therefore `RSS_max` degenerates
4. and MODQN is left to rank beams mostly through learned Q differences driven by access history, beam load, and global rewards rather than by per-beam channel separation

## Why More Episodes Are A Weak Next Step

Existing repo evidence already shows that long training is risky:

1. `artifacts/run-9000/anomaly-review.md` documents a real late-training collapse beginning around episode `5700`
2. that review explicitly recommends artifact review before more retraining

Relevant lines:

- `artifacts/run-9000/anomaly-review.md:15-23`
- `artifacts/run-9000/anomaly-review.md:108-117`

Combined interpretation:

1. the current baseline semantics already compress the decision surface
2. the reward geometry is already throughput-dominant
3. a previous long run already showed late collapse

Therefore, more episodes are unlikely to reveal a qualitatively different result unless the contract itself changes.

## Classification Summary

Keep as baseline contract if the goal is:

1. disclosed paper-faithful baseline, including known anomalies
2. stable downstream export/replay truth for `ntn-sim-core`
3. comparison against the repo's already frozen assumption surface

Promote to reopen candidate if the goal is:

1. beam-level channel discrimination that better matches an intuitive multi-beam problem
2. physically lossy atmospheric attenuation in the primary run
3. a meaningful `RSS_max` comparator
4. stronger fairness/load-balance credit assignment

## Recommended Next Step

Do not start another `9000`-episode run by default.

Instead:

1. treat this note as the current read-only decision surface
2. decide which of the following is the first intentional reopen target:
   - `beam-level channel / eligibility semantics`
   - `atmospheric sign / reward geometry`
   - `RSS comparator semantics`
   - `global-r3 credit assignment`
3. only after one of those reopen targets is explicitly chosen should a new training track be considered
