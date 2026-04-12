# Notebook → `queue-rl/` Refactor Blueprint

This document maps the code currently in `Final_Trial3_assignment_actor.ipynb` to your proposed multi-file project layout.

## 1) What goes where

### `src/envs/queue_env.py`
Move all environment dynamics and wrappers here:
- `QueueEnv` class.
- Reward wrappers and transforms:
  - `RewardScaleWrapper`
  - `LostArrivalPenaltyWrapper`
  - `QueueMeanAsReward`
- Environment builder helpers:
  - `make_queue_env`
  - `make_eval_env`

**From notebook:** environment and wrappers are currently in code cells 1 and 6.

---

### `src/models/critic.py`
Move centralized value network code here:
- `CTDECritic`

**From notebook:** code cell 2.

---

### `src/models/policy.py`
Move policy and actor stack here:
- `RobotQueueAttentionActor`
- `QueuePolicyNet_Attn`
- `QueuePPOPolicy`

Keep policy-specific masking/sampling logic in this file so it stays cohesive with SB3 policy integration.

**From notebook:** code cell 2.

---

### `src/baselines/weighted_greedy.py`
Move heuristic baselines and assignment utilities here:
- `make_weighted_greedy_no_collision`
- `greedy_longest_no_collision`
- helper set metrics used for policy comparisons:
  - `_occupied_queue_set`
  - `_service_queue_set`
  - `_set_jaccard`

**From notebook:** code cells 3 and 4.

---

### `src/evaluation/eval.py`
Move all evaluation orchestration and statistics code here:
- `mean_ci95`
- `_run_one_episode`
- `rollout_collect_mean_q_parallel`
- `fmt_mean_pm`
- `compare_policies_mean_q`
- `ParallelQueueEvalCallback`
- plotting helpers:
  - `plot_queue_evolution`
  - `plot_cap_count_evolution`

**From notebook:** code cells 4, 6, and 10.

---

### `src/training/train.py`
Keep this as the experiment entrypoint for training:
- configuration loading
- env/model construction
- callback wiring (`CallbackList`, `RolloutInfoMean`, `ParallelQueueEvalCallback`)
- PPO `.learn(...)`
- best-model save path logic

Also move this callback class here (or to `src/utils/logging.py` if you want callback-only organization):
- `RolloutInfoMean`

**From notebook:** code cells 6–9 (the procedural training/eval-run sections).

---

### `src/utils/seed.py`
Move reproducibility helpers here:
- global seed utility wrapping:
  - `random.seed`
  - `numpy.random.seed`
  - `torch.manual_seed`
  - SB3 `set_random_seed`

The notebook imports seeding utilities but does not centralize them yet; create this as a new utility.

---

### `src/utils/logging.py`
Move logging-facing helper code here:
- metric formatting helpers such as `fmt_mean_pm`
- optional tensorboard/log-dir naming utilities
- optional callback classes focused on logging only

If you prefer cleaner separation, keep numerical evaluation functions in `evaluation/eval.py` and only keep logging/output formatting utilities here.

---

### `configs/*.yaml`
Represent all notebook hardcoded experiment knobs in config files:
- system size: `M`, `N`
- queue parameters: `arrival_params`, `queue_cost_weights`, `max_queue_length`
- barrier settings: `barrier_start`, `barrier_power`, etc.
- episode/training sizes: `max_steps_per_run`, `total_timesteps`, `n_steps`, `batch_size`, `learning_rate`
- evaluation settings: `T`, `num_episodes`, `eval_freq`, `seed`

Use `base.yaml` for defaults and scenario overlays (`scenario_small.yaml`, `scenario_large.yaml`).

---

### `scripts/run_train.sh` and `scripts/run_eval.sh`
Shell wrappers should only parse args and call Python module entrypoints, e.g.:
- `python -m src.training.train --config configs/scenario_small.yaml`
- `python -m src.evaluation.eval --config configs/scenario_small.yaml --checkpoint ...`

---

### `tests/*`
Suggested mapping:
- `tests/test_env.py`
  - reset/step contracts
  - collision constraints
  - service/switching semantics
- `tests/test_policy_shapes.py`
  - actor logits and masking shape checks
  - policy output dimensions for varying `M,N`
- `tests/test_eval_smoke.py`
  - tiny 1–2 episode integration smoke test for eval pipeline

## 2) Keep out of `src/`
Notebook-only items should not be migrated as-is:
- ad-hoc cell execution state
- one-off plotting display calls embedded in training flow
- direct “load best model from path if exists” snippets (convert to CLI options)

## 3) Recommended extraction order
1. `src/envs/queue_env.py`
2. `src/models/{critic.py,policy.py}`
3. `src/baselines/weighted_greedy.py`
4. `src/evaluation/eval.py`
5. `src/training/train.py`
6. tests + scripts + configs

This order minimizes breakage because env and model APIs stabilize first.

## 4) Import graph target
- `training/train.py` imports envs, models, baselines, evaluation callback.
- `evaluation/eval.py` imports env + baselines + trained policy loading.
- `models/*` do not import training/evaluation modules.
- `envs/*` does not import training module.

That keeps the package acyclic and easier to test.
