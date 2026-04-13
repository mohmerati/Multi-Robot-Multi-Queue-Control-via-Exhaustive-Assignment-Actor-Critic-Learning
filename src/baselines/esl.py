import numpy as np
from typing import Dict, Callable


def make_weighted_esl(
    arrival_rates: np.ndarray,
    queue_cost_weights: np.ndarray,
) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
    """
    Factory for a deterministic exhaustive weighted-greedy baseline.

    Switching index:
        I_i = w_i * q_i + lambda_i

    Policy structure:
      Pass 1 (exhaustive service):
        - If robot r is currently at queue i and q_i > 0, then it stays and serves,
          provided i has not already been reserved by an earlier robot.

      Pass 2 (switching only for non-serving robots):
        - Assign each remaining robot to a distinct queue with largest weighted index.
        - Tie handling is deterministic:
            (a) if the robot's current queue is among the best candidates, stay there;
            (b) otherwise choose the lowest-index best candidate.

    Edge-case handling:
      - Validates shapes and finiteness.
      - Clips tiny negative queue values to 0.
      - If all queues are empty, a robot does not steal another robot's current location
        unless it has no alternative candidate.
      - If candidate set becomes empty, the robot stays at its current location.
    """
    arrival_rates = np.asarray(arrival_rates, dtype=np.float32).reshape(-1)
    queue_cost_weights = np.asarray(queue_cost_weights, dtype=np.float32).reshape(-1)

    if arrival_rates.ndim != 1 or queue_cost_weights.ndim != 1:
        raise ValueError("arrival_rates and queue_cost_weights must be 1D arrays.")
    if arrival_rates.shape != queue_cost_weights.shape:
        raise ValueError("arrival_rates and queue_cost_weights must have the same shape.")
    if not np.all(np.isfinite(arrival_rates)):
        raise ValueError("arrival_rates contains non-finite values.")
    if not np.all(np.isfinite(queue_cost_weights)):
        raise ValueError("queue_cost_weights contains non-finite values.")
    if np.any(queue_cost_weights < 0):
        raise ValueError("queue_cost_weights must be nonnegative.")

    def policy(obs: Dict[str, np.ndarray]) -> np.ndarray:
        if "robots" not in obs or "queues" not in obs:
            raise KeyError("obs must contain keys 'robots' and 'queues'.")

        robots = np.asarray(obs["robots"], dtype=np.int64).reshape(-1)      # [M]
        queues = np.asarray(obs["queues"], dtype=np.float32).reshape(-1)    # [N]

        M = robots.shape[0]
        N = queues.shape[0]

        if arrival_rates.shape[0] != N:
            raise ValueError(
                f"arrival_rates has length {arrival_rates.shape[0]}, but obs['queues'] has length {N}."
            )
        if queue_cost_weights.shape[0] != N:
            raise ValueError(
                f"queue_cost_weights has length {queue_cost_weights.shape[0]}, but obs['queues'] has length {N}."
            )
        if not np.all(np.isfinite(queues)):
            raise ValueError("obs['queues'] contains non-finite values.")
        if np.any((robots < 0) | (robots >= N)):
            raise ValueError("obs['robots'] contains an invalid queue index.")

        if M == 0:
            return np.empty((0,), dtype=np.int64)

        # Robustness against tiny numerical negatives
        queues = np.maximum(queues, 0.0)

        # Weighted switching index
        scores = queue_cost_weights * queues + arrival_rates

        actions = robots.copy()
        reserved_q = set()
        assigned_r = set()

        all_empty = bool(np.all(queues == 0.0))
        occupied_now = set(map(int, robots.tolist()))

        # ------------------------------------------------------------
        # Pass 1: exhaustive local service
        # ------------------------------------------------------------
        for r in range(M):
            loc = int(robots[r])
            if queues[loc] > 0.0 and loc not in reserved_q:
                actions[r] = loc
                reserved_q.add(loc)
                assigned_r.add(r)

        # ------------------------------------------------------------
        # Pass 2: assign distinct destinations by weighted index
        # ------------------------------------------------------------
        tol = 1e-12

        for r in range(M):
            if r in assigned_r:
                continue

            loc = int(robots[r])

            # Feasible unreserved destinations
            pos_cands = [j for j in range(N) if j not in reserved_q]

            # Safety fallback
            if not pos_cands:
                actions[r] = loc
                continue

            # When all queues are empty, avoid pointless stealing/swapping:
            # do not send robot r to another robot's current location unless forced.
            if all_empty:
                filtered = [j for j in pos_cands if (j == loc) or (j not in occupied_now)]
                if filtered:
                    pos_cands = filtered

            cand_scores = scores[pos_cands]
            best_score = float(np.max(cand_scores))

            # All candidates attaining the maximum score (within tolerance)
            best_cands = [j for j in pos_cands if scores[j] >= best_score - tol]

            # Deterministic tie handling:
            # prefer staying if current location is optimal; otherwise lower index.
            if loc in best_cands:
                j_best = loc
            else:
                j_best = min(best_cands)

            actions[r] = int(j_best)
            reserved_q.add(int(j_best))

        return actions.astype(np.int64)

    return policy

def make_esl(
    arrival_rates: np.ndarray,
) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
    """
    Deterministic unweighted ESL baseline.

    Rule:
      1) If a robot is at a nonempty queue, stay and serve there.
      2) Otherwise assign remaining robots to distinct longest available queues.
         Tie-break:
           - prefer staying if current queue is among best
           - otherwise prefer higher arrival rate
           - otherwise lower queue index
    """
    arrival_rates = np.asarray(arrival_rates, dtype=np.float32).reshape(-1)

    if arrival_rates.ndim != 1:
        raise ValueError("arrival_rates must be a 1D array.")
    if not np.all(np.isfinite(arrival_rates)):
        raise ValueError("arrival_rates contains non-finite values.")

    def policy(obs: Dict[str, np.ndarray]) -> np.ndarray:
        if "robots" not in obs or "queues" not in obs:
            raise KeyError("obs must contain keys 'robots' and 'queues'.")

        robots = np.asarray(obs["robots"], dtype=np.int64).reshape(-1)
        queues = np.asarray(obs["queues"], dtype=np.float32).reshape(-1)

        M = robots.shape[0]
        N = queues.shape[0]

        if arrival_rates.shape[0] != N:
            raise ValueError(
                f"arrival_rates has length {arrival_rates.shape[0]}, but obs['queues'] has length {N}."
            )
        if not np.all(np.isfinite(queues)):
            raise ValueError("obs['queues'] contains non-finite values.")
        if np.any((robots < 0) | (robots >= N)):
            raise ValueError("obs['robots'] contains an invalid queue index.")

        if M == 0:
            return np.empty((0,), dtype=np.int64)

        queues = np.maximum(queues, 0.0)

        actions = robots.copy()
        reserved_q = set()
        assigned_r = set()

        all_empty = bool(np.all(queues == 0.0))
        occupied_now = set(map(int, robots.tolist()))

        # Pass 1: exhaustive local service
        for r in range(M):
            loc = int(robots[r])
            if queues[loc] > 0.0 and loc not in reserved_q:
                actions[r] = loc
                reserved_q.add(loc)
                assigned_r.add(r)

        # Pass 2: assign distinct longest remaining queues
        for r in range(M):
            if r in assigned_r:
                continue

            loc = int(robots[r])
            pos_cands = [j for j in range(N) if j not in reserved_q]

            if not pos_cands:
                actions[r] = loc
                continue

            if all_empty:
                filtered = [j for j in pos_cands if (j == loc) or (j not in occupied_now)]
                if filtered:
                    pos_cands = filtered

            best_len = float(np.max(queues[pos_cands]))
            best_cands = [j for j in pos_cands if queues[j] == best_len]

            if loc in best_cands:
                j_best = loc
            else:
                max_arr = float(np.max(arrival_rates[best_cands]))
                arr_best = [j for j in best_cands if arrival_rates[j] == max_arr]
                j_best = min(arr_best)

            actions[r] = int(j_best)
            reserved_q.add(int(j_best))

        return actions.astype(np.int64)

    return policy