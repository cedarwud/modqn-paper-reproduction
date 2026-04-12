"""RSS-max baseline evaluation."""

from __future__ import annotations

import numpy as np

from ..algorithms.modqn import EvalSummary
from ..env.step import StepEnvironment


def evaluate_rss_max(
    env: StepEnvironment,
    *,
    evaluation_seed_set: tuple[int, ...],
    scalarization_weights: tuple[float, float, float],
    episode: int = -1,
    evaluation_every_episodes: int = 0,
) -> EvalSummary:
    """Evaluate the fixed RSS-max policy over the configured eval seeds."""
    if not evaluation_seed_set:
        raise ValueError("evaluation_seed_set must be non-empty for evaluation")

    rows: list[dict[str, float]] = []
    for seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        states, masks, _diag = env.reset(env_rng, mobility_rng)
        ep_reward = np.zeros(3, dtype=np.float64)
        ep_handovers = 0

        for _step_idx in range(env.config.steps_per_episode):
            actions = np.zeros(len(masks), dtype=np.int32)
            for uid, mask in enumerate(masks):
                valid = np.where(mask.mask)[0]
                if len(valid) == 0:
                    actions[uid] = 0
                    continue
                snr = states[uid].channel_quality.copy()
                snr[~mask.mask] = -np.inf
                actions[uid] = int(np.argmax(snr))

            result = env.step(actions, env_rng)
            for rw in result.rewards:
                reward_vec = np.array(
                    [rw.r1_throughput, rw.r2_handover, rw.r3_load_balance],
                    dtype=np.float64,
                )
                ep_reward += reward_vec
                if rw.r2_handover < 0:
                    ep_handovers += 1

            if result.done:
                break
            states = result.user_states
            masks = result.action_masks

        avg_reward = ep_reward / max(env.config.num_users, 1)
        rows.append(
            {
                "scalar_reward": float(np.dot(scalarization_weights, avg_reward)),
                "r1_mean": float(avg_reward[0]),
                "r2_mean": float(avg_reward[1]),
                "r3_mean": float(avg_reward[2]),
                "total_handovers": float(ep_handovers),
            }
        )

    def mean_std(key: str) -> tuple[float, float]:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        return float(np.mean(values)), float(np.std(values))

    scalar_mean, scalar_std = mean_std("scalar_reward")
    r1_mean, r1_std = mean_std("r1_mean")
    r2_mean, r2_std = mean_std("r2_mean")
    r3_mean, r3_std = mean_std("r3_mean")
    handover_mean, handover_std = mean_std("total_handovers")

    return EvalSummary(
        episode=episode,
        evaluation_every_episodes=evaluation_every_episodes,
        eval_seeds=tuple(int(seed) for seed in evaluation_seed_set),
        mean_scalar_reward=scalar_mean,
        std_scalar_reward=scalar_std,
        mean_r1=r1_mean,
        std_r1=r1_std,
        mean_r2=r2_mean,
        std_r2=r2_std,
        mean_r3=r3_mean,
        std_r3=r3_std,
        mean_total_handovers=handover_mean,
        std_total_handovers=handover_std,
    )
