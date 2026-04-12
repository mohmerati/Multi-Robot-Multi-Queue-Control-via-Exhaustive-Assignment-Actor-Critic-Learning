# Multi-Robot Multi-Queue Control via Exhaustive Assignment Actor-Critic Learning

This repository contains research code for online control of multi-robot, multi-queue systems with stochastic arrivals and switching delays. The project studies a discrete-time setting in which each location can host at most one robot per slot, service takes one time step, switching between queues incurs a one-slot delay, and arrivals follow heterogeneous Bernoulli processes. The goal is to minimize discounted holding cost over time.

The main contribution of this code is a structure-aware reinforcement learning policy based on an **Exhaustive-Assignment Actor-Critic (EA-AC)** architecture. Instead of learning unconstrained joint actions for all robots, the policy enforces exhaustive service by construction: robots at nonempty queues continue serving, while only idle robots are reassigned. This reduces the effective action space and allows the policy to focus on the nontrivial part of the scheduling problem.

---

## Problem Setting

We consider a system with:

- `M` mobile robots (servers)
- `N` spatially distributed queues (task locations), with `M <= N`
- independent Bernoulli task arrivals at each queue
- deterministic unit service time
- deterministic one-step switching delay between queues
- a one-robot-per-queue feasibility constraint

At each decision step, the controller determines where idle robots should go next. The objective is to minimize the discounted sum of queue lengths.

---

## Method Overview

The repository implements a PPO-based actor-critic approach tailored to the structure of the problem.

### Key ideas

- **Exhaustive service restriction**  
  A robot at a nonempty queue continues serving until that queue becomes empty.

- **Assignment-only actor**  
  The actor learns only the reassignment of idle robots.

- **Sequential masked action generation**  
  Idle robots are assigned one at a time with feasibility masks to prevent collisions and duplicate destination assignments.

- **Centralized critic**  
  The critic evaluates the global congestion state using pooled queue and robot representations.

- **Greedy baseline for comparison**  
  The code includes an exhaustive greedy baseline based on the ESL principle, along with weighted variants for comparison.

---

## Repository Contents

- `Final_Trial3_assignment_actor.ipynb`  
  Main notebook containing:
  - the `QueueEnv` Gymnasium environment
  - actor and critic network definitions
  - custom PPO policy logic
  - greedy / weighted-greedy baselines
  - arrival-rate generation utilities
  - training callbacks
  - parallel evaluation code
  - result plotting utilities

- `Multi-Robot Multi-Queue Control via Exhaustive Assignment Actor-Critic Learning.pdf`  
  Paper describing the mathematical formulation, learning architecture, and experimental results.

- `docs/notebook_to_package_map.md`  
  Refactor blueprint mapping notebook components into a production-style `queue-rl/` package layout.

You may later want to split the notebook into modules such as `env.py`, `policy.py`, `baselines.py`, and `eval.py`, but for now the notebook-based structure is perfectly acceptable for a research release.

---

## Requirements

This project uses Python with the following main packages:

- `numpy`
- `torch`
- `gymnasium`
- `stable-baselines3`
- `matplotlib`
- `tensorboard` (optional, for logging)

A typical installation would look like:

```bash
pip install numpy torch gymnasium stable-baselines3 matplotlib tensorboard
