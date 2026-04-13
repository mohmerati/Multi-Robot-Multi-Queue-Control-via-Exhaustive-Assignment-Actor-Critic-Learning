import numpy as np
from typing import Any, Dict
from src.baselines import make_weighted_esl
from src.evaluation.parallel_eval import rollout_collect_mean_q_parallel

def mean_ci95(x: np.ndarray):
    """
    Returns (mean, half_width, (lo, hi)) for 95% CI across samples in x.
    Uses Student-t if SciPy available; else normal approx.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    m = float(x.mean())
    if n == 1:
        return m, np.nan, (np.nan, np.nan)
    s = float(x.std(ddof=1))
    se = s / np.sqrt(n)

    crit = 1.959963984540054  # ~N(0,1) 97.5th percentile

    hw = crit * se
    return m, hw, (m - hw, m + hw)

def fmt_mean_pm(ci_dict, digits=4):
    return f"{ci_dict['mean']:.{digits}f} ± {ci_dict['half_width']:.{digits}f}"

def compare_policies_mean_q(
    env_kwargs: Dict[str, Any],
    seed: int = 123,
    T: int = 10_000,
    num_episodes: int = 10,
    model: "PPO" = None,
):
    
    if model is None:
        raise ValueError("compare_policies_mean_q requires a trained PPO model.")
    
    dis_fact = 0.99

    greedy_longest_no_collision = make_weighted_esl(
        arrival_rates=np.asarray(env_kwargs["arrival_params"], dtype=np.float32),
        queue_cost_weights=np.asarray(env_kwargs["queue_cost_weights"], dtype=np.float32),
    )

    greedy_res = rollout_collect_mean_q_parallel(
        env_kwargs, seed=seed, T=T, num_episodes=num_episodes,
        policy_fn_or_model=greedy_longest_no_collision,
        discount_factor=dis_fact,
        n_jobs=None,   # or set e.g. n_jobs=8
    )
    ppo_res = rollout_collect_mean_q_parallel(
        env_kwargs, seed=seed, T=T, num_episodes=num_episodes,
        policy_fn_or_model=model,
        discount_factor=dis_fact,
        n_jobs=None,
    )

    print("=== Comparison of Greedy Baseline vs PPO Policy ===")
    print("environemt settings:")
    print(env_kwargs["M"], "robots, ", env_kwargs["N"], "queues, arrival rates:", env_kwargs["arrival_params"])

    print("\n=== Mean queue length (episode means) ===")
    print(f"Greedy baseline:  overall mean = {fmt_mean_pm(greedy_res['ci95']['overall_mean'], 6)}")
    print(f"PPO policy:       overall mean = {fmt_mean_pm(ppo_res['ci95']['overall_mean'], 6)}")

    print("\n=== Total discounted cost (episode means) ===")
    print(f"Greedy baseline:  total cost = {fmt_mean_pm(greedy_res['ci95']['total_cost'], 6)}")
    print(f"PPO policy:       total cost = {fmt_mean_pm(ppo_res['ci95']['total_cost'], 6)}")

    print("\n=== Action fractions (episode means) ===")
    print(f"Greedy baseline:  serve = {fmt_mean_pm(greedy_res['ci95']['serve_fraction'], 4)}")
    print(f"PPO policy:       serve = {fmt_mean_pm(ppo_res['ci95']['serve_fraction'], 4)}")
    print(f"Greedy baseline:  idle = {fmt_mean_pm(greedy_res['ci95']['idle_fraction'], 4)}")
    print(f"PPO policy:       idle = {fmt_mean_pm(ppo_res['ci95']['idle_fraction'], 4)}")
    print(f"Greedy baseline:  switch = {fmt_mean_pm(greedy_res['ci95']['switch_fraction'], 4)}")
    print(f"PPO policy:       switch = {fmt_mean_pm(ppo_res['ci95']['switch_fraction'], 4)}")
    
    print("\n=== Queue-cap occupancy ===")
    print(f"Greedy baseline:  mean capped queues/step = {fmt_mean_pm(greedy_res['ci95']['mean_cap_count'])}, "
        f"steps with any capped queue = {fmt_mean_pm(greedy_res['ci95']['any_cap_fraction'])}")

    print(f"PPO policy:       mean capped queues/step = {fmt_mean_pm(ppo_res['ci95']['mean_cap_count'])}, "
        f"steps with any capped queue = {fmt_mean_pm(ppo_res['ci95']['any_cap_fraction'])}")

    print(f"PPO policy:       mean capped queues/step = {ppo_res['mean_cap_count']:.4f} "
        f"(95% CI [{ppo_res['ci95']['mean_cap_count']['lo']:.4f}, {ppo_res['ci95']['mean_cap_count']['hi']:.4f}]), "
        f"steps with any capped queue = {ppo_res['any_cap_fraction']:.4f} "
        f"(95% CI [{ppo_res['ci95']['any_cap_fraction']['lo']:.4f}, {ppo_res['ci95']['any_cap_fraction']['hi']:.4f}])")
    
    print("\n=== Similarity to greedy on visited states ===")
    print(f"Greedy baseline:  occupied-set exact = {fmt_mean_pm(greedy_res['ci95']['occ_exact_match_fraction'])}, "
        f"occupied-set Jaccard = {fmt_mean_pm(greedy_res['ci95']['occ_jaccard_mean'])}")

    print(f"PPO policy:       occupied-set exact = {fmt_mean_pm(ppo_res['ci95']['occ_exact_match_fraction'])}, "
        f"occupied-set Jaccard = {fmt_mean_pm(ppo_res['ci95']['occ_jaccard_mean'])}")

    print(f"Greedy baseline:  service-set exact  = {fmt_mean_pm(greedy_res['ci95']['serv_exact_match_fraction'])}, "
        f"service-set Jaccard = {fmt_mean_pm(greedy_res['ci95']['serv_jaccard_mean'])}")

    print(f"PPO policy:       service-set exact  = {fmt_mean_pm(ppo_res['ci95']['serv_exact_match_fraction'])}, "
        f"service-set Jaccard = {fmt_mean_pm(ppo_res['ci95']['serv_jaccard_mean'])}")
    
    return greedy_res, ppo_res