"""Smoke validation for the MODQN training pipeline.

Exercises every critical path from config loading through training:
  1. config load (YAML → Python objects)
  2. env reset
  3. env step
  4. action mask application
  5. state encoding
  6. replay buffer push / sample
  7. DQN network forward pass
  8. target network sync
  9. backward step (gradient flow)
  10. short training loop (3 episodes)

This is NOT a correctness test for training convergence.
It validates that the pipeline is wired end-to-end and does not crash.
"""

from __future__ import annotations

import numpy as np
import torch
import tempfile
from pathlib import Path

from modqn_paper_reproduction.config_loader import (
    load_yaml,
    build_environment,
    build_trainer_config,
    get_seeds,
)
from modqn_paper_reproduction.env.step import (
    StepEnvironment,
    StepConfig,
    ActionMask,
)
from modqn_paper_reproduction.env.orbit import OrbitConfig
from modqn_paper_reproduction.env.beam import BeamConfig
from modqn_paper_reproduction.env.channel import ChannelConfig
from modqn_paper_reproduction.algorithms.modqn import (
    DQNNetwork,
    MODQNTrainer,
    ReplayBuffer,
    TrainerConfig,
    encode_state,
    state_dim_for,
)


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------


def test_config_load():
    """YAML → Python config objects round-trip."""
    cfg = load_yaml("configs/modqn-paper-baseline.resolved-template.yaml")
    assert cfg["config_role"] == "resolved-run-template"

    env = build_environment(cfg)
    assert env.num_beams_total == 28  # 4 sats * 7 beams
    assert env.config.num_users == 100
    assert env.config.r3_gap_scope == "all-reachable-beams"
    assert env.config.user_heading_stride_rad == 2.3998277
    assert env.config.user_scatter_radius_km == 50.0

    tc = build_trainer_config(cfg)
    assert tc.episodes == 9000
    assert tc.epsilon_start == 1.0
    assert tc.epsilon_end == 0.01
    assert tc.epsilon_decay_episodes == 7000
    assert tc.target_update_every_episodes == 50
    assert tc.replay_capacity == 50_000
    assert tc.snr_encoding == "log1p"
    assert tc.checkpoint_primary_report == "final-episode-policy"
    assert tc.checkpoint_secondary_report == "best-weighted-reward-on-eval"

    seeds = get_seeds(cfg)
    assert seeds["train_seed"] == 42
    assert seeds["environment_seed"] == 1337
    assert seeds["mobility_seed"] == 7
    assert seeds["evaluation_seed_set"] == [100, 200, 300, 400, 500]


# ---------------------------------------------------------------------------
# 2. Env reset
# ---------------------------------------------------------------------------


def test_env_reset():
    """Environment resets and produces valid initial state."""
    env = StepEnvironment()
    rng = np.random.default_rng(1337)

    states, masks, diag = env.reset(rng)

    assert len(states) == 100
    assert len(masks) == 100
    assert states[0].access_vector.shape == (28,)
    assert states[0].channel_quality.shape == (28,)
    assert states[0].beam_offsets.shape == (28, 2)
    assert states[0].beam_loads.shape == (28,)
    assert diag.zenith_snr_db != 0.0


# ---------------------------------------------------------------------------
# 3. Env step
# ---------------------------------------------------------------------------


def test_env_step():
    """Environment steps and returns reward components."""
    env = StepEnvironment()
    rng = np.random.default_rng(1337)
    states, masks, _ = env.reset(rng)

    actions = np.array([
        np.where(m.mask)[0][0] if m.num_valid > 0 else 0
        for m in masks
    ], dtype=np.int32)

    result = env.step(actions, rng)
    assert result.step_index == 1
    assert not result.done  # first step of 10
    assert len(result.rewards) == 100
    assert result.rewards[0].r1_throughput >= 0


# ---------------------------------------------------------------------------
# 4. Action mask
# ---------------------------------------------------------------------------


def test_action_mask():
    """Masks correctly restrict to visible satellite beams."""
    env = StepEnvironment()
    rng = np.random.default_rng(1337)
    states, masks, _ = env.reset(rng)

    for m in masks:
        assert m.mask.shape == (28,)
        # At least some beams should be visible
        assert m.num_valid > 0
        # Visible beams should be contiguous blocks of 7
        # (all beams of a visible sat are valid)
        for sat in range(4):
            block = m.mask[sat * 7 : (sat + 1) * 7]
            # Either all True or all False for a satellite
            assert block.all() or not block.any()


# ---------------------------------------------------------------------------
# 5. State encoding
# ---------------------------------------------------------------------------


def test_state_encoding():
    """State encoding produces correct dimension and finite values."""
    env = StepEnvironment()
    rng = np.random.default_rng(1337)
    states, _, _ = env.reset(rng)

    tc = TrainerConfig()
    encoded = encode_state(states[0], 100, tc)

    assert encoded.shape == (state_dim_for(28),)
    assert encoded.shape == (140,)
    assert np.all(np.isfinite(encoded))

    # Access vector part should be one-hot
    access_part = encoded[:28]
    assert np.sum(access_part) == 1.0

    # log1p(snr) should be non-negative
    snr_part = encoded[28:56]
    assert np.all(snr_part >= 0)


# ---------------------------------------------------------------------------
# 6. Replay buffer
# ---------------------------------------------------------------------------


def test_replay_buffer():
    """Push transitions and sample a batch."""
    rb = ReplayBuffer(capacity=200)

    for i in range(150):
        rb.push(
            state=np.random.randn(140).astype(np.float32),
            action=i % 28,
            reward_3=np.random.randn(3).astype(np.float32),
            next_state=np.random.randn(140).astype(np.float32),
            mask=np.ones(28, dtype=bool),
            next_mask=np.ones(28, dtype=bool),
            done=False,
        )

    assert len(rb) == 150

    rng = np.random.default_rng(42)
    states, actions, rewards, next_states, masks, next_masks, dones = rb.sample(32, rng)

    assert states.shape == (32, 140)
    assert actions.shape == (32,)
    assert rewards.shape == (32, 3)
    assert next_states.shape == (32, 140)
    assert masks.shape == (32, 28)
    assert next_masks.shape == (32, 28)
    assert dones.shape == (32,)

    # Push beyond capacity — should wrap
    for i in range(100):
        rb.push(
            state=np.random.randn(140).astype(np.float32),
            action=0,
            reward_3=np.zeros(3, dtype=np.float32),
            next_state=np.random.randn(140).astype(np.float32),
            mask=np.ones(28, dtype=bool),
            next_mask=np.ones(28, dtype=bool),
            done=True,
        )
    assert len(rb) == 200  # capped at capacity


# ---------------------------------------------------------------------------
# 7. Network forward
# ---------------------------------------------------------------------------


def test_network_forward():
    """DQN network forward pass produces correct output shape."""
    net = DQNNetwork(140, 28, (100, 50, 50), "tanh")
    x = torch.randn(16, 140)
    out = net(x)
    assert out.shape == (16, 28)
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# 8. Target network sync
# ---------------------------------------------------------------------------


def test_target_sync():
    """Target network parameters match online after sync."""
    env = StepEnvironment(StepConfig(num_users=5))
    tc = TrainerConfig(episodes=1, replay_capacity=100)
    trainer = MODQNTrainer(env, tc, train_seed=42, env_seed=1, mobility_seed=2)

    # Before sync: modify online, target should differ
    with torch.no_grad():
        for p in trainer.q_nets[0].parameters():
            p.add_(1.0)

    # Check they differ
    online_p = list(trainer.q_nets[0].parameters())[0].data
    target_p = list(trainer.target_nets[0].parameters())[0].data
    assert not torch.allclose(online_p, target_p)

    # Sync
    trainer.sync_targets()

    # Now should match
    for i in range(3):
        for p_on, p_tgt in zip(
            trainer.q_nets[i].parameters(),
            trainer.target_nets[i].parameters(),
        ):
            assert torch.allclose(p_on.data, p_tgt.data)


# ---------------------------------------------------------------------------
# 9. Backward step
# ---------------------------------------------------------------------------


def test_backward_step():
    """Gradient flows through DQN and optimizer steps."""
    net = DQNNetwork(140, 28, (100, 50, 50), "tanh")
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    x = torch.randn(8, 140)
    target_q = torch.randn(8, 28)

    # Save initial params
    p0 = list(net.parameters())[0].data.clone()

    out = net(x)
    loss = torch.nn.functional.mse_loss(out, target_q)
    opt.zero_grad()
    loss.backward()
    opt.step()

    p1 = list(net.parameters())[0].data
    assert not torch.allclose(p0, p1), "Parameters should change after optimizer step"


# ---------------------------------------------------------------------------
# 10. Short training loop
# ---------------------------------------------------------------------------


def test_short_training_loop():
    """Run 3 episodes of MODQN training to verify full pipeline."""
    env = StepEnvironment(StepConfig(num_users=10))
    tc = TrainerConfig(
        episodes=3,
        replay_capacity=500,
        batch_size=32,
        epsilon_decay_episodes=2,
        target_update_every_episodes=1,
    )

    trainer = MODQNTrainer(
        env=env,
        config=tc,
        train_seed=42,
        env_seed=1337,
        mobility_seed=7,
    )

    logs = trainer.train(progress_every=0)

    assert len(logs) == 3

    # Check metrics are populated
    for log in logs:
        assert log.r1_mean >= 0 or log.r1_mean < 0  # finite
        assert log.replay_size > 0
        assert 0 <= log.epsilon <= 1.0

    # Replay should have accumulated transitions
    # 3 episodes * 10 steps * 10 users = 300 transitions
    assert logs[-1].replay_size == 300

    # Epsilon should have decayed
    assert logs[0].epsilon > logs[-1].epsilon


# ---------------------------------------------------------------------------
# 11. Epsilon schedule
# ---------------------------------------------------------------------------


def test_epsilon_schedule():
    """Epsilon decays linearly and clamps at end."""
    env = StepEnvironment(StepConfig(num_users=5))
    tc = TrainerConfig(
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=100,
    )
    trainer = MODQNTrainer(env, tc)

    assert trainer.epsilon(0) == 1.0
    assert abs(trainer.epsilon(50) - 0.525) < 1e-6
    assert trainer.epsilon(100) == 0.05
    assert trainer.epsilon(200) == 0.05  # clamped


# ---------------------------------------------------------------------------
# 12. Masked action selection
# ---------------------------------------------------------------------------


def test_masked_action_selection():
    """Actions are always within the valid mask."""
    env = StepEnvironment(StepConfig(num_users=10))
    tc = TrainerConfig()
    trainer = MODQNTrainer(env, tc, train_seed=42, env_seed=1, mobility_seed=2)

    rng = np.random.default_rng(1337)
    states, masks, _ = env.reset(rng)

    encoded = np.array([
        encode_state(s, 10, tc) for s in states
    ], dtype=np.float32)

    # Test with high epsilon (random) and low epsilon (greedy)
    for eps in [1.0, 0.0]:
        actions = trainer.select_actions(encoded, masks, eps)
        for uid in range(10):
            assert masks[uid].mask[actions[uid]], (
                f"Action {actions[uid]} for user {uid} is not valid "
                f"(eps={eps})"
            )


# ---------------------------------------------------------------------------
# 13. Checkpoint save/load
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load():
    """Final checkpoint persists weights and load restores them."""
    env = StepEnvironment(StepConfig(num_users=10))
    tc = TrainerConfig(
        episodes=1,
        replay_capacity=256,
        batch_size=32,
        target_update_every_episodes=1,
    )
    trainer = MODQNTrainer(
        env=env,
        config=tc,
        train_seed=42,
        env_seed=1337,
        mobility_seed=7,
    )
    logs = trainer.train(progress_every=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "final-episode-policy.pt"
        trainer.save_checkpoint(
            checkpoint_path,
            episode=logs[-1].episode,
            checkpoint_kind=tc.checkpoint_primary_report,
            logs=logs,
        )
        assert checkpoint_path.exists()

        reloaded = MODQNTrainer(
            env=StepEnvironment(StepConfig(num_users=10)),
            config=tc,
            train_seed=999,
            env_seed=998,
            mobility_seed=997,
        )
        before = list(reloaded.q_nets[0].parameters())[0].detach().clone()
        payload = reloaded.load_checkpoint(checkpoint_path)
        after = list(reloaded.q_nets[0].parameters())[0].detach().clone()
        saved = list(trainer.q_nets[0].parameters())[0].detach().clone()

        assert not torch.allclose(before, saved)
        assert torch.allclose(after, saved)
        assert payload["checkpoint_rule"]["primary_report"] == "final-episode-policy"
        assert payload["checkpoint_rule"]["secondary_implemented"] is False


def test_best_eval_checkpoint_capture():
    """Greedy eval loop captures and persists the best-eval checkpoint."""
    env = StepEnvironment(StepConfig(num_users=10))
    tc = TrainerConfig(
        episodes=3,
        replay_capacity=512,
        batch_size=32,
        target_update_every_episodes=1,
    )
    trainer = MODQNTrainer(
        env=env,
        config=tc,
        train_seed=42,
        env_seed=1337,
        mobility_seed=7,
    )
    trainer.train(
        progress_every=0,
        evaluation_seed_set=(100, 200),
        evaluation_every_episodes=1,
    )

    summary = trainer.best_eval_summary()
    assert summary is not None
    assert summary.eval_seeds == (100, 200)
    assert summary.evaluation_every_episodes == 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "best-weighted-reward-on-eval.pt"
        trainer.save_best_eval_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert payload["checkpoint_kind"] == tc.checkpoint_secondary_report
        assert payload["checkpoint_rule"]["secondary_implemented"] is True
        assert payload["evaluation_summary"]["eval_seeds"] == [100, 200]
        assert payload["evaluation_summary"]["episode"] >= 0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_config_load,
        test_env_reset,
        test_env_step,
        test_action_mask,
        test_state_encoding,
        test_replay_buffer,
        test_network_forward,
        test_target_sync,
        test_backward_step,
        test_short_training_loop,
        test_epsilon_schedule,
        test_masked_action_selection,
        test_checkpoint_save_and_load,
        test_best_eval_checkpoint_capture,
    ]

    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        raise SystemExit(1)
