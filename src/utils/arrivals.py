import numpy as np

def generate_arrival_params_load_new(
    N: int,
    M: int,
    block_sum: float = 0.8,          # total load / M
    step: float = 0.05,
    seed: int | None = None,
    lam_max: float = 1.0,
    alpha_dirichlet: float = 1.0,
    enforce_nonneg: bool = True,
    lam_min: float = 0.05,           # NEW
) -> np.ndarray:
    """
    Generate lambda[0..N-1] on a step grid such that

        sum(lambda) / M == block_sum

    with per-queue bounds
        lam_min <= lambda_i <= cap := min(lam_max, block_sum).

    Smaller alpha_dirichlet => more heterogeneous.
    """
    if N <= 0 or M <= 0:
        raise ValueError("N and M must be positive integers.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if lam_max <= 0:
        raise ValueError("lam_max must be positive.")
    if block_sum < 0:
        raise ValueError("block_sum must be nonnegative.")
    if alpha_dirichlet <= 0:
        raise ValueError("alpha_dirichlet must be positive.")
    if lam_min < 0:
        raise ValueError("lam_min must be nonnegative.")

    rng = np.random.default_rng(seed)

    cap = float(min(lam_max, block_sum))
    total_target = float(block_sum) * float(M)

    if lam_min > cap + 1e-12:
        raise ValueError(
            f"Infeasible: lam_min={lam_min:.4f} exceeds cap={cap:.4f}."
        )

    if total_target < N * lam_min - 1e-12:
        raise ValueError(
            f"Infeasible under lower bound λ_i ≥ {lam_min:.4f}: "
            f"target sum = {total_target:.4f} is below N*lam_min = {N*lam_min:.4f}."
        )

    if total_target > N * cap + 1e-12:
        raise ValueError(
            f"Infeasible under upper bound λ_i ≤ {cap:.4f}: "
            f"target sum = {total_target:.4f} exceeds N*cap = {N*cap:.4f}."
        )

    total_target_grid = round(total_target / step) * step
    if abs(total_target_grid - total_target) > 1e-6:
        raise ValueError(
            f"Target total {total_target:.6f} is not a multiple of step={step}. "
            f"Nearest grid total is {total_target_grid:.6f}."
        )
    total_target = total_target_grid

    lam_min_grid = round(lam_min / step) * step
    if abs(lam_min_grid - lam_min) > 1e-6:
        raise ValueError(
            f"lam_min={lam_min:.6f} is not a multiple of step={step}."
        )
    lam_min = lam_min_grid

    # Reserve lam_min for every queue, then distribute the remainder.
    residual_total = total_target - N * lam_min
    residual_cap = cap - lam_min

    if residual_total < -1e-12:
        raise ValueError("Infeasible residual total after enforcing lam_min.")
    if residual_cap < -1e-12:
        raise ValueError("Infeasible residual cap after enforcing lam_min.")

    # Special case: all queues must be exactly lam_min
    if residual_total <= 1e-12:
        lam = np.full(N, lam_min, dtype=np.float64)
        return lam.astype(np.float32)

    alpha_vec = np.full(N, float(alpha_dirichlet), dtype=np.float64)
    w = rng.dirichlet(alpha=alpha_vec)
    lam_res = w * residual_total

    lam_res = np.round(lam_res / step) * step
    lam_res = np.clip(lam_res, 0.0, residual_cap)

    cur = float(lam_res.sum())
    ticks = int(round((residual_total - cur) / step))

    for _ in range(10_000_000):
        if ticks == 0:
            break

        if ticks > 0:
            slack_up = np.where(lam_res <= residual_cap - step + 1e-12)[0]
            if slack_up.size == 0:
                raise RuntimeError(
                    "Cannot increase further to meet target sum under the cap."
                )
            k = min(ticks, slack_up.size)
            idx = rng.choice(slack_up, size=k, replace=False)
            lam_res[idx] += step
            ticks -= k
        else:
            slack_dn = np.where(lam_res >= step - 1e-12)[0]
            if slack_dn.size == 0:
                raise RuntimeError("Cannot decrease further to meet target sum.")
            k = min(-ticks, slack_dn.size)
            idx = rng.choice(slack_dn, size=k, replace=False)
            lam_res[idx] -= step
            ticks += k

    lam = lam_min + lam_res

    if enforce_nonneg:
        lam = np.clip(lam, lam_min, cap)

    if abs(lam.sum() - total_target) > 1e-6:
        raise RuntimeError(
            f"Failed to match target sum. sum={lam.sum():.6f}, target={total_target:.6f}"
        )

    if np.any(lam < lam_min - 1e-12):
        raise RuntimeError("Lower bound violated.")
    if np.any(lam > cap + 1e-12):
        raise RuntimeError("Upper bound violated.")

    return lam.astype(np.float32)