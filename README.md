# Multi-Robot Multi-Queue Control via Exhaustive Assignment Actor-Critic Learning

This repository contains research code for structure-aware reinforcement learning in multi-robot, multi-queue scheduling systems with stochastic arrivals and switching delays. The project studies a discrete-time setting in which each queue can host at most one robot per slot, service takes one time step, switching between queues incurs a one-step delay, and arrivals follow heterogeneous Bernoulli processes. The objective is to minimize discounted holding cost over time.

The main learning contribution is an **Exhaustive-Assignment Actor-Critic (EA-AC)** architecture. Instead of learning unconstrained joint actions for all robots, the policy enforces exhaustive service by construction: robots at nonempty queues continue serving, while only idle robots are reassigned. This reduces the effective action space and focuses learning on the nontrivial part of the scheduling problem.

---

## Problem Setting

We consider a system with:

- `M` mobile robots (servers)
- `N` spatially distributed queues, with `M <= N`
- independent Bernoulli task arrivals at each queue
- deterministic unit service time
- deterministic one-step switching delay between queues
- a one-robot-per-queue feasibility constraint

At each time step, the controller decides where robots should go next, subject to feasibility and exhaustive-service structure. The performance objective is the discounted sum of queue lengths over time.

---

## Method Overview

The repository implements a PPO-based actor-critic framework specialized to this structured scheduling problem.

### Main ideas

- **Exhaustive service restriction**  
  A robot at a nonempty queue keeps serving until that queue becomes empty.

- **Assignment-only actor**  
  The actor only decides the reassignment of idle robots.

- **Sequential masked action generation**  
  Idle robots are assigned one at a time with feasibility masks to avoid collisions and duplicate assignments.

- **Centralized critic**  
  The critic evaluates the global congestion state using pooled queue and robot features.

- **Greedy baselines**  
  The code includes exhaustive greedy baselines, including weighted ESL variants, for comparison.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Final_Trial3_assignment_actor.ipynb
├── scripts/
│   ├── __init__.py
│   └── run_eval.py
├── src/
│   ├── __init__.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── esl.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── queue_env.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── eval.py
│   │   ├── parallel_eval.py
│   │   └── stats.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── queue_ppo_policy.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── config_smoke.py
│   │   └── train.py
│   └── utils/
│       ├── __init__.py
│       └── arrivals.py
└── outputs/
    ├── checkpoints/
    └── tensorboard/
````

### Folder descriptions

* `src/envs/`
  Gymnasium environment for the multi-robot multi-queue system.

* `src/models/`
  Actor, critic, and SB3-compatible custom PPO policy wrapper.

* `src/baselines/`
  Deterministic ESL and weighted-ESL baseline policies.

* `src/training/`
  Training script and experiment configuration files.

* `src/evaluation/`
  Parallel rollout evaluation, policy comparison utilities, and summary statistics.

* `src/utils/`
  Shared utilities, including arrival-rate generation.

* `scripts/`
  Entry-point scripts, such as full evaluation of trained checkpoints.

* `notebooks/`
  Original research notebook retained as reference.

* `outputs/`
  Saved checkpoints, TensorBoard logs, and generated outputs.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Requirements

Main dependencies include:

* `numpy`
* `torch`
* `gymnasium`
* `stable-baselines3`
* `matplotlib`
* `tensorboard`

You can install everything with:

```bash
pip install -r requirements.txt
```

---

## Configuration

Training and evaluation are driven by configuration files in `src/training/`.

### Main config

* `src/training/config.py`
  Full experiment configuration.

### Smoke test config

* `src/training/config_smoke.py`
  Small configuration for fast pipeline checks.

Typical parameters defined there include:

* system size: `M`, `N`
* load and arrival-generation parameters
* queue cap and horizon
* PPO hyperparameters
* evaluation settings
* output directories

---

## Running a Smoke Test

Before launching a long training run, it is recommended to test the full pipeline on a small instance.

For example, using the smoke config:

```bash
python -m src.training.train
```

If `train.py` is currently pointed to `config_smoke.py`, this should execute a short training run and verify that:

* the environment builds correctly
* the PPO policy initializes
* rollouts run without crashing
* outputs are written to the expected directories

---

## Training

The main training entrypoint is:

```bash
python -m src.training.train
```

This script:

1. builds arrival-rate parameters from the config
2. constructs the vectorized training environment
3. initializes the custom PPO policy
4. trains the EA-AC model
5. logs metrics to TensorBoard
6. optionally runs evaluation callbacks and saves the best checkpoint

### Training outputs

By default, outputs are stored under:

* `outputs/tensorboard/`
* `outputs/checkpoints/`

Each run uses a directory name derived from the experiment settings, such as:

```text
M3_N9_u0.7_Qmax30_lr0.0007_nsteps32_batch32
```

---

## Evaluation

The main evaluation entrypoint is:

```bash
python -m scripts.run_eval
```

This script:

1. reconstructs the evaluation scenario from the config
2. loads a trained checkpoint
3. evaluates the learned PPO policy against the weighted ESL baseline
4. computes summary metrics and confidence intervals
5. generates plots for queue evolution and capped-queue behavior

### Expected checkpoint location

Evaluation expects a saved model under:

```text
outputs/checkpoints/<run_name>/best_model.zip
```

or another checkpoint file such as `final_model.zip`, depending on how training was configured.

---

## Metrics Reported

The evaluation code reports and compares:

* mean queue length
* discounted total cost
* serve / idle / switch fractions
* capped-queue statistics
* similarity to the greedy baseline on visited states

The rollout code also supports trajectory-level summaries such as:

* time-mean queue evolution
* queue-cap evolution
* occupied-set similarity
* service-set similarity

---

## Baselines

The repository includes deterministic greedy baselines in `src/baselines/esl.py`.

### Included baselines

* **ESL**
  Exhaustive Serve Longest with deterministic tie-breaking.

* **Weighted ESL**
  Uses a weighted switching index of the form:

```text
I_i = w_i q_i + \lambda_i
```

where:

* `q_i` is the queue length
* `w_i` is the queue cost weight
* `\lambda_i` is the arrival rate

These baselines are used during evaluation for comparison against the learned PPO policy.

---

## Original Notebook

The original notebook is retained in:

```text
notebooks/Final_Trial3_assignment_actor.ipynb
```

It contains the earlier notebook-based implementation of:

* the environment
* actor and critic definitions
* custom PPO policy logic
* baselines
* evaluation code
* plotting utilities

The refactored modular code in `src/` is intended to replace the notebook for regular use, but the notebook remains useful as a reference and historical record of the research workflow.

---

## Reproducibility Notes

Exact equality between notebook-based and refactored results should not always be expected, especially for PPO training, due to:

* stochastic optimization
* vectorized rollout ordering
* checkpoint selection differences
* evaluation parallelism
* software/runtime differences

However, the refactored code preserves the main modeling, training, and evaluation logic, and is intended to produce consistent behavior under the same configuration.

---

## Typical Workflow

### 1. Run a smoke test

```bash
python -m src.training.train
```

### 2. Run a full training experiment

Edit `src/training/config.py`, then run:

```bash
python -m src.training.train
```

### 3. Evaluate the trained model

```bash
python -m scripts.run_eval
```

### 4. Inspect logs

```bash
tensorboard --logdir outputs/tensorboard
```

---

## Notes

* `outputs/` is typically ignored by Git.
* The virtual environment `.venv/` should not be committed.
* Package imports assume commands are run from the repository root.

---

## Citation

If you use this code in your work, please cite the associated paper:

**Merati, M., Ahmad, H. M. S., Li, W., Castanón, D.**
*Multi-Robot Multi-Queue Control via Exhaustive Assignment Actor-Critic Learning.*

You may also cite the related structural-control paper:

**Merati, M., Castanón, D.**
*Exhaustive-Serve-Longest Control for Multi-robot Scheduling Systems.*

---

## Contact

For questions about the code or the project, please open an issue on the repository.

```

A few optional improvements would make this even better:
- add a short “Quick Start” near the top,
- include one example checkpoint path,
- and add a small section called “Common Pitfalls” for import-path and virtual-environment issues.
```
