import numpy as np
from typing import Dict, Tuple, Any
from src.envs.queue_env import QueueEnv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from src.baselines import make_weighted_esl
from src.evaluation.eval import mean_ci95

def _occupied_queue_set(action: np.ndarray) -> set:
    """
    Unordered set of queues selected/occupied by the action.
    Since your policies enforce no-collision, this is the set of chosen destinations.
    """
    action = np.asarray(action, dtype=np.int64)
    return set(map(int, action.tolist()))

def _service_queue_set(obs: Dict[str, np.ndarray], action: np.ndarray) -> set:
    """
    Set of queues that are served immediately at this state under the action.
    In your env, queue i is served now iff some robot currently at i stays and Q_i > 0.
    """
    robots = np.asarray(obs["robots"], dtype=np.int64)
    queues = np.asarray(obs["queues"], dtype=np.float32)
    action = np.asarray(action, dtype=np.int64)

    served = set()
    for r in range(robots.shape[0]):
        loc = int(robots[r])
        if int(action[r]) == loc and queues[loc] > 0:
            served.add(loc)
    return served

def _set_jaccard(A: set, B: set) -> float:
    """
    Jaccard similarity between two sets.
    Defined as 1 when both are empty.
    """
    union = A | B
    if len(union) == 0:
        return 1.0
    return len(A & B) / len(union)

def _run_one_episode(
    ep: int,
    env_kwargs: Dict[str, Any],
    base_seed: int,
    T: int,
    discount_factor: float,
    policy_kind: str,   # "callable" or "ppo"
) -> Tuple[np.ndarray, float, float, float, float, np.ndarray, float, float, float, float, float, float]:
    """
    Runs exactly ONE episode (length T) and returns:
        (means, serve_frac, idle_frac, switch_frac, total_cost, cap_counts,
        mean_cap_count, any_cap_fraction, occ_exact_match_fraction, occ_jaccard_mean,
        serv_exact_match_fraction, serv_jaccard_mean)

    Minimal changes vs your rollout_collect_mean_q inner loop, but:
        - seeds independently per episode => parallel-friendly
    """
    # Optional: avoid CPU oversubscription when many processes spawn torch threads
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    env = QueueEnv(**env_kwargs)

    # Per-episode independent seed (parallel-safe)
    ep_seed = int(base_seed + ep)
    obs, _ = env.reset(seed=ep_seed)

    greedy_policy = make_weighted_esl(
        arrival_rates=np.asarray(env_kwargs["arrival_params"], dtype=np.float32),
        queue_cost_weights=np.asarray(
            env_kwargs.get("queue_cost_weights", np.ones(env.N, dtype=np.float32)),
            dtype=np.float32,
        ),
    )

    means = []
    serve_count = 0
    idle_count = 0
    switch_count = 0

    total_cost = 0.0
    disc = 1.0

    cap_counts = []
    any_cap_steps = 0

    # --- similarity-to-greedy accumulators
    occ_exact_hits = 0
    occ_jacc_sum = 0.0
    serv_exact_hits = 0
    serv_jacc_sum = 0.0

    # Access PPO model via global to avoid pickling issues under fork
    # (see wrapper below where we set _GLOBAL_POLICY)
    global _GLOBAL_POLICY
    policy_obj = _GLOBAL_POLICY

    for t in range(T):
        if policy_kind == "ppo":
            a = policy_obj.predict(obs, deterministic=True)[0]
        else:
            a = policy_obj(obs)

        # --- compare against greedy on the SAME state
        greedy_a = greedy_policy(obs)

        occ_pol = _occupied_queue_set(a)
        occ_greedy = _occupied_queue_set(greedy_a)

        serv_pol = _service_queue_set(obs, a)
        serv_greedy = _service_queue_set(obs, greedy_a)

        occ_exact_hits += int(occ_pol == occ_greedy)
        occ_jacc_sum += _set_jaccard(occ_pol, occ_greedy)

        serv_exact_hits += int(serv_pol == serv_greedy)
        serv_jacc_sum += _set_jaccard(serv_pol, serv_greedy)

        # Count serve vs idle vs switch
        for r in range(env.M):
            loc = int(obs["robots"][r])
            if int(a[r]) == loc:
                if obs["queues"][loc] > 0:
                    serve_count += 1
                else:
                    idle_count += 1
            else:
                switch_count += 1

        obs, reward, terminated, truncated, info = env.step(a)

        means.append(float(info["queues_mean"]))

        total_cost += disc * (-reward)
        disc *= discount_factor

        cap_counts.append(float(info["cap_count"]))
        any_cap_steps += int(info["any_cap"])

        # validity checks kept (same as yours)
        if not (np.issubdtype(a.dtype, np.integer) and a.shape == (env.M,)):
            raise RuntimeError(f"Invalid action shape/dtype from policy: shape {a.shape}, dtype {a.dtype}")
        if np.any(a < 0) or np.any(a >= env.N):
            raise RuntimeError(f"Invalid action values from policy: {a}")

        # If the env ends mid-episode, reset WITHOUT reseeding
        if terminated or truncated:
            obs, _ = env.reset(seed=None)

    means = np.asarray(means, dtype=np.float64)   # shape [T]
    cap_counts = np.asarray(cap_counts, dtype=np.float64)

    total_actions = serve_count + idle_count + switch_count
    serve_frac = serve_count / total_actions if total_actions > 0 else 0.0
    idle_frac = idle_count / total_actions if total_actions > 0 else 0.0
    switch_frac = switch_count / total_actions if total_actions > 0 else 0.0
    total_cost = float(total_cost)

    # NEW
    mean_cap_count = float(cap_counts.mean())
    any_cap_fraction = float(any_cap_steps / T)

    occ_exact_match_fraction = float(occ_exact_hits / T)
    occ_jaccard_mean = float(occ_jacc_sum / T)

    serv_exact_match_fraction = float(serv_exact_hits / T)
    serv_jaccard_mean = float(serv_jacc_sum / T)

    return (
        means,
        serve_frac,
        idle_frac,
        switch_frac,
        total_cost,
        cap_counts,
        mean_cap_count,
        any_cap_fraction,
        occ_exact_match_fraction,
        occ_jaccard_mean,
        serv_exact_match_fraction,
        serv_jaccard_mean,
    )

def rollout_collect_mean_q_parallel(
    env_kwargs: Dict[str, Any],
    seed: int,
    T: int,
    policy_fn_or_model,
    discount_factor: float,
    num_episodes: int = 10,
    n_jobs: int = None,
) -> Dict[str, Any]:
    """
    Parallel version of rollout_collect_mean_q.
    Returns the same keys you already use downstream (means/overall_mean/etc + ci95),
    but episode_means_series is omitted (would be huge + expensive to ship between processes).
    """
    global _GLOBAL_POLICY
    _GLOBAL_POLICY = policy_fn_or_model

    policy_kind = "ppo" if hasattr(policy_fn_or_model, "predict") else "callable"

    ctx = mp.get_context("fork")  # same style as your second script

    # Run episodes in parallel
    with ProcessPoolExecutor(mp_context=ctx, max_workers=n_jobs) as ex:
        results = list(
            ex.map(
                _run_one_episode,
                range(num_episodes),
                [env_kwargs] * num_episodes,
                [seed] * num_episodes,
                [T] * num_episodes,
                [discount_factor] * num_episodes,
                [policy_kind] * num_episodes,
            )
        )

    (
        ep_means_series,
        ep_serve_frac,
        ep_idle_frac,
        ep_switch_frac,
        ep_total_cost,
        ep_cap_counts_series,
        ep_mean_cap_count,
        ep_any_cap_fraction,
        ep_occ_exact_match_fraction,
        ep_occ_jaccard_mean,
        ep_serv_exact_match_fraction,
        ep_serv_jaccard_mean,
    ) = zip(*results)

    ep_means_series = np.stack(ep_means_series, axis=0)   # [num_episodes, T]
    ep_cap_counts_series = np.stack(ep_cap_counts_series, axis=0)  # [E, T]

    ep_serve_frac = np.asarray(ep_serve_frac, dtype=np.float64)
    ep_idle_frac = np.asarray(ep_idle_frac, dtype=np.float64)
    ep_switch_frac = np.asarray(ep_switch_frac, dtype=np.float64)
    ep_total_cost = np.asarray(ep_total_cost, dtype=np.float64)

    ep_overall_mean = ep_means_series.mean(axis=1)        # [num_episodes]

    ep_mean_cap_count = np.asarray(ep_mean_cap_count, dtype=np.float64)
    ep_any_cap_fraction = np.asarray(ep_any_cap_fraction, dtype=np.float64)

    ep_occ_exact_match_fraction = np.asarray(ep_occ_exact_match_fraction, dtype=np.float64)
    ep_occ_jaccard_mean = np.asarray(ep_occ_jaccard_mean, dtype=np.float64)
    ep_serv_exact_match_fraction = np.asarray(ep_serv_exact_match_fraction, dtype=np.float64)
    ep_serv_jaccard_mean = np.asarray(ep_serv_jaccard_mean, dtype=np.float64)

    mean_m, mean_hw, mean_ci = mean_ci95(ep_overall_mean)
    serve_m, serve_hw, serve_ci = mean_ci95(ep_serve_frac)
    idle_m, idle_hw, idle_ci = mean_ci95(ep_idle_frac)
    switch_m, switch_hw, switch_ci = mean_ci95(ep_switch_frac)
    cost_m, cost_hw, cost_ci = mean_ci95(ep_total_cost)

    # New
    cap_count_m, cap_count_hw, cap_count_ci = mean_ci95(ep_mean_cap_count)
    any_cap_m, any_cap_hw, any_cap_ci = mean_ci95(ep_any_cap_fraction)

    occ_exact_m, occ_exact_hw, occ_exact_ci = mean_ci95(ep_occ_exact_match_fraction)
    occ_jacc_m, occ_jacc_hw, occ_jacc_ci = mean_ci95(ep_occ_jaccard_mean)
    serv_exact_m, serv_exact_hw, serv_exact_ci = mean_ci95(ep_serv_exact_match_fraction)
    serv_jacc_m, serv_jacc_hw, serv_jacc_ci = mean_ci95(ep_serv_jaccard_mean)
    
    time_mean = ep_means_series.mean(axis=0)              # [T]
    time_std = ep_means_series.std(axis=0, ddof=1) if num_episodes > 1 else np.zeros(T)
    time_se = time_std / np.sqrt(num_episodes)
    crit = 1.959963984540054
    time_ci_lo = time_mean - crit * time_se
    time_ci_hi = time_mean + crit * time_se

    # New
    cap_time_mean = ep_cap_counts_series.mean(axis=0)
    cap_time_std = ep_cap_counts_series.std(axis=0, ddof=1) if num_episodes > 1 else np.zeros(T)
    cap_time_se = cap_time_std / np.sqrt(num_episodes)
    crit = 1.959963984540054
    cap_time_ci_lo = cap_time_mean - crit * cap_time_se
    cap_time_ci_hi = cap_time_mean + crit * cap_time_se

    return {
        # NOTE: no episode_means_series to keep IPC light
        "episode_means_series": ep_means_series,   # [num_episodes, T]
        "episode_overall_mean": ep_overall_mean,
        "episode_serve_fraction": ep_serve_frac,
        "episode_idle_fraction": ep_idle_frac,
        "episode_switch_fraction": ep_switch_frac,
        "episode_total_cost": ep_total_cost,

        "means": ep_overall_mean,
        "overall_mean": float(ep_overall_mean.mean()),
        "serve_fraction": float(ep_serve_frac.mean()),
        "idle_fraction": float(ep_idle_frac.mean()),
        "switch_fraction": float(ep_switch_frac.mean()),
        "total_cost": float(ep_total_cost.mean()),

        "episode_cap_counts_series": ep_cap_counts_series,
        "episode_mean_cap_count": ep_mean_cap_count,
        "episode_any_cap_fraction": ep_any_cap_fraction,

        "mean_cap_count": float(ep_mean_cap_count.mean()),
        "any_cap_fraction": float(ep_any_cap_fraction.mean()),

        "episode_occ_exact_match_fraction": ep_occ_exact_match_fraction,
        "episode_occ_jaccard_mean": ep_occ_jaccard_mean,
        "episode_serv_exact_match_fraction": ep_serv_exact_match_fraction,
        "episode_serv_jaccard_mean": ep_serv_jaccard_mean,

        "occ_exact_match_fraction": float(ep_occ_exact_match_fraction.mean()),
        "occ_jaccard_mean": float(ep_occ_jaccard_mean.mean()),
        "serv_exact_match_fraction": float(ep_serv_exact_match_fraction.mean()),
        "serv_jaccard_mean": float(ep_serv_jaccard_mean.mean()),

        "cap_time_mean": cap_time_mean,
        "cap_time_ci_lo": cap_time_ci_lo,
        "cap_time_ci_hi": cap_time_ci_hi,

        "time_mean": time_mean,
        "time_ci_lo": time_ci_lo,
        "time_ci_hi": time_ci_hi,

        "ci95": {
            "overall_mean": {"mean": mean_m, "half_width": mean_hw, "lo": mean_ci[0], "hi": mean_ci[1]},
            "serve_fraction": {"mean": serve_m, "half_width": serve_hw, "lo": serve_ci[0], "hi": serve_ci[1]},
            "idle_fraction": {"mean": idle_m, "half_width": idle_hw, "lo": idle_ci[0], "hi": idle_ci[1],},
            "switch_fraction": {"mean": switch_m, "half_width": switch_hw, "lo": switch_ci[0], "hi": switch_ci[1]},
            "total_cost": {"mean": cost_m, "half_width": cost_hw, "lo": cost_ci[0], "hi": cost_ci[1]},
            "mean_cap_count": {"mean": cap_count_m, "half_width": cap_count_hw, "lo": cap_count_ci[0], "hi": cap_count_ci[1]},
            "any_cap_fraction": {"mean": any_cap_m, "half_width": any_cap_hw, "lo": any_cap_ci[0], "hi": any_cap_ci[1]},
            "occ_exact_match_fraction": {"mean": occ_exact_m, "half_width": occ_exact_hw, "lo": occ_exact_ci[0], "hi": occ_exact_ci[1]},
            "occ_jaccard_mean": {"mean": occ_jacc_m, "half_width": occ_jacc_hw, "lo": occ_jacc_ci[0], "hi": occ_jacc_ci[1]},
            "serv_exact_match_fraction": {"mean": serv_exact_m, "half_width": serv_exact_hw, "lo": serv_exact_ci[0], "hi": serv_exact_ci[1]},
            "serv_jaccard_mean": {"mean": serv_jacc_m, "half_width": serv_jacc_hw, "lo": serv_jacc_ci[0], "hi": serv_jacc_ci[1]},
        }
    }