## DQN ablation of PER vs uniform vs online replay, parameter noise vs ε-greedy

This repository studies two independent design choices in Deep Q-Networks:

1. **Replay strategy**: Prioritized Experience Replay (PER) vs uniform replay vs online updates (no buffer)
2. **Exploration strategy**: parameter-space noise vs ε-greedy

Experiments run on Gymnasium's `LunarLander-v3`. Each run logs training metrics, saves checkpoints, plots reward curves, and records a rollout video whenever a new best policy appears.

---

## Methods

### DQN

The agent approximates the action-value function Q(s, a) with a neural network and minimizes the TD error against a target network:

```
δ = r + γ · max_a' Q_target(s', a') − Q(s, a)
```

The two ablation axes below change **which transitions** the network learns from and **how the agent explores**; everything else is held fixed.

### Replay strategies

Selected with `replay_mode`.

| Mode | Description |
|---|---|
| `per` | **Prioritized Experience Replay.** Transitions are sampled in proportion to their priority p_i = \|δ_i\|, so surprising transitions are replayed more often. Importance-sampling weights correct the resulting bias, with β annealed toward 1 over training. |
| `uniform` | Transitions are sampled uniformly at random from the replay buffer (standard DQN). |
| `online` | No replay buffer: the network is updated on each transition as it arrives. Serves as a no-replay baseline. |

PER sampling probability and importance-sampling weight:

```
P(i) = p_i^α / Σ_k p_k^α
w_i  = (N · P(i))^(−β) / max_j w_j
```

Priorities are refreshed with the new |δ| after each update.

### Exploration strategies

Selected with `exploration_mode`.

| Mode | Description |
|---|---|
| `param_noise` | **Parameter-space noise.** Gaussian noise (scale `noise_scale`) is added to the network's parameters to produce a perturbed policy that is held fixed for an episode. Exploration is state-dependent and consistent within an episode, unlike per-step action noise. |
| `epsilon_greedy` | With probability ε take a random action, otherwise act greedily. Standard DQN baseline. |
| `none` | Always act greedily with respect to the current Q-network. |

---

## Ablation design

The ablation script runs a 2 × 2 factorial comparison of replay strategy × exploration noise:

| | No parameter noise | Parameter noise |
|---|---|---|
| **Uniform replay** | uniform / no noise | uniform / param noise |
| **PER** | PER / no noise | PER / param noise |

This isolates the effect of each component and shows whether they interact. Online replay and ε-greedy are available as additional baselines through config overrides (see [Training](#training)).

---

## Environment

`LunarLander-v3` (Gymnasium Box2D):

- Observation: 8-dimensional continuous state
- Actions: 4 discrete (no-op, left engine, main engine, right engine)
- Reward shaping encourages stable landing and penalizes crashes and fuel use

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training

### Default training

```bash
python scripts/train.py --config configs/default.yaml
```

### Override config values from the CLI

```bash
python scripts/train.py --config configs/default.yaml \
  --set replay_mode=uniform exploration_mode=none
```

Common overrides:

| Key | Values |
|---|---|
| `replay_mode` | `per` \| `uniform` \| `online` |
| `exploration_mode` | `param_noise` \| `epsilon_greedy` \| `none` |
| `noise_scale` | e.g. `0.1` |
| `num_episodes` | e.g. `800` |

---

## Running the ablation

```bash
python scripts/ablation.py \
  --base configs/default.yaml \
  --ablation configs/ablation.yaml
```

Each configuration gets its own run folder under `experiments/`, containing:

- `config.yaml`: the resolved configuration for the run
- `models/`: checkpoints, including `best_model.pth`
- plots of the reward curves

---

## Evaluation

```bash
python scripts/evaluate.py \
  --config experiments/<run>/config.yaml \
  --checkpoint experiments/<run>/models/best_model.pth \
  --episodes 20
```

---

## Video recording

```bash
python scripts/record_video.py \
  --config experiments/<run>/config.yaml \
  --checkpoint experiments/<run>/models/best_model.pth \
  --out_dir assets/videos \
  --name best_lunarlander_dqn
```

---

## Testing

```bash
pytest -q
```

---

## License

MIT (see `LICENSE`).
