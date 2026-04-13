import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces

class QueueEnv(gym.Env):
    """
    Multi-robot, multi-queue env with switching delay = 1.
    Actions per robot: which queue to go to next (including current).
    Observation: dict(robots: positions, queues: lengths).
    Reward: -sum(queues).
    """
    metadata = {"render_modes": []}

    def __init__(self,
             M: int = 2,
             N: int = 4,
             arrival_params=None,
             queue_cost_weights=None,
             seed: Optional[int] = None,
             lambda_collision: float = 0.0,
             max_steps_per_run: Optional[int] = None,
             max_queue_length: int = 50,
             barrier_start: float = 0.8,
             alpha: float = 20,
             ):
        super().__init__()
        self.M, self.N = M, N
        self.lambda_collision = float(lambda_collision)
        self.max_steps_per_run = max_steps_per_run                 # <<< NEW
        self.barrier_start = barrier_start * max_queue_length      # <<< NEW
        self.alpha = alpha

        if arrival_params is None:
            arrival_params = np.full(N, 0.10, dtype=np.float32)
        self.arrival_params = np.asarray(arrival_params, dtype=np.float32)

        if queue_cost_weights is None:
            queue_cost_weights = np.ones(N, dtype=np.float32)
        self.queue_cost_weights = np.asarray(queue_cost_weights, dtype=np.float32)

        assert self.queue_cost_weights.shape == (N,), \
            f"queue_cost_weights must have shape ({N},), got {self.queue_cost_weights.shape}"

        # RNG and step counter
        self._seed = None
        self.rng = np.random.default_rng(seed)
        self._step_count = 0                                       # <<< NEW

        # State
        self._queues = np.zeros(self.N, dtype=np.float32)
        self._robots = np.arange(self.M, dtype=np.int64)

        # Spaces
        self.observation_space = spaces.Dict({
            "robots": spaces.MultiDiscrete([N] * M),
            "queues": spaces.Box(low=0, high=max_queue_length, shape=(N,), dtype=np.float32)
        })
        self.action_space = spaces.MultiDiscrete([N] * M)

    def _get_obs(self):
        return {"robots": self._robots.copy(), "queues": self._queues.copy()}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Gymnasium seeding API
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        self._step_count = 0                                       # <<< reset counter
        self._queues.fill(0.0)
        self._robots = np.arange(self.M, dtype=np.int64)
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def _resolve_collisions(self, intended_pos):
        # Reserve all current positions initially
        final_pos = self._robots.copy()
        taken = set(final_pos.tolist())

        projected = 0
        for r in range(self.M):
            tgt = int(intended_pos[r])
            cur = int(self._robots[r])

            if tgt == cur:
                # Stayer: already reserved; nothing to do
                continue

            if tgt not in taken:
                # Move succeeds: free old spot, claim target
                final_pos[r] = tgt
                taken.remove(cur)
                taken.add(tgt)
            else:
                # Move fails: stay; count projection
                projected += 1
                # final_pos[r] already equals cur; 'taken' unchanged

        return final_pos, projected

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.int64)
        assert action.shape == (self.M,), f"Expected action shape {(self.M,)}, got {action.shape}"

        # --- snapshot BEFORE
        queues_before = self._queues.copy()
        robots_before = self._robots.copy()

        # --- reward from current queues (pre-cost)
        qsmean = float(self._queues.mean())
        # # --- NEW: weighted queue cost and diagnostics
        # qsmean = float(np.dot(self.queue_cost_weights, self._queues) / self.N)

        # penalty_cap = self.alpha * np.sum(np.maximum(0.0, (self._queues - self.barrier_start) / (self.observation_space["queues"].high[0] - self.barrier_start)) ** 2)
        # reward = -float(self._queues.sum()) - penalty_cap
        weighted_cost = float(np.dot(self.queue_cost_weights, queues_before))
        reward = -weighted_cost

        action = np.asarray(action, dtype=np.int64)
        stay_mask = (action == robots_before).astype(np.int64)
        switch_mask = (action != robots_before).astype(np.int64)

        # --- serve (record who actually served)
        served_mask = np.zeros(self.M, dtype=np.int64)
        for r in range(self.M):
            loc = int(self._robots[r])
            if action[r] == loc and self._queues[loc] > 0:
                self._queues[loc] -= 1.0
                served_mask[r] = 1

        empty_stay_mask = np.zeros(self.M, dtype=np.int64)
        for r in range(self.M):
            loc = int(robots_before[r])
            if stay_mask[r] == 1 and queues_before[loc] <= 0:
                empty_stay_mask[r] = 1

        # --- switch intentions
        intended = self._robots.copy()
        for r in range(self.M):
            intended[r] = action[r]

        # --- apply collision resolution
        new_pos, num_proj = self._resolve_collisions(intended)
        self._robots = new_pos

        # --- arrivals
        p = self.arrival_params
        arrivals = self.rng.binomial(n=1, p=p, size=self.N).astype(np.float32)

        qmax = float(self.observation_space["queues"].high[0])
        room = np.maximum(0.0, qmax - self._queues)
        admitted = np.minimum(arrivals, room)
        dropped = arrivals - admitted

        self._queues = self._queues + admitted

        # --- collision penalty
        if self.lambda_collision > 0 and num_proj > 0:
            reward -= self.lambda_collision * float(num_proj)

        # --- truncation logic
        self._step_count += 1
        truncated = (
            self.max_steps_per_run is not None
            and self._step_count >= self.max_steps_per_run
        )
        terminated = False
        obs = self._get_obs()
        # print(f"queues_mean: {qsmean}, collisions: {int(num_proj)}")  # <<< DEBUG PRINT
        
        cap_count = int((self._queues >= self.observation_space["queues"].high[0] - 1e-8).sum())
        any_cap = int(cap_count > 0)

        info: Dict[str, Any] = {
            "queues_mean": qsmean,
            "collisions_count": int(num_proj),

            # diagnostics
            "stay_count": int(stay_mask.sum()),
            "switch_count": int(switch_mask.sum()),
            "served_count": int(served_mask.sum()),
            "empty_stay_count": int(empty_stay_mask.sum()),
            "cap_count": cap_count,
            "any_cap": any_cap,

            # truncation diagnostics
            "lost_arrivals_count": int(dropped.sum()),
            "weighted_lost_arrivals": float(np.dot(self.queue_cost_weights, dropped)),
        }

        # --- TRACE: rich info for debugging/analysis
        info: Dict[str, Any] = {
            "queues_before": queues_before,
            "robots_before": robots_before,
            "action_array": action.copy(),
            "served_mask": served_mask,
            "intended_positions": intended,
            "robots_after": self._robots.copy(),
            "arrivals": arrivals,
            "num_projections": int(num_proj),
            "reward": reward,
            "truncated": truncated,
        }
        
        return obs, reward, terminated, truncated, info

    # def get_action_mask(self, exhaustive: bool = True) -> np.ndarray:
    #     """
    #     Returns an (M, N) binary mask.
    #     mask[r, j] = 1 if robot r is allowed to choose queue j.

    #     If exhaustive=True:
    #     - a robot sitting on a nonempty queue can only stay there
    #     - otherwise, robot can choose any queue except those currently occupied
    #         by other robots; staying at its own current location is allowed
    #     """
    #     mask = np.ones((self.M, self.N), dtype=np.int8)

    #     occupied = set(int(x) for x in self._robots.tolist())

    #     for r in range(self.M):
    #         cur = int(self._robots[r])
    #         qcur = float(self._queues[cur])

    #         if exhaustive and qcur > 0:
    #             mask[r, :] = 0
    #             mask[r, cur] = 1
    #             continue

    #         # idle robot: cannot choose queues occupied by other robots
    #         for j in occupied:
    #             if j != cur:
    #                 mask[r, j] = 0

    #         # staying at current queue is always allowed
    #         mask[r, cur] = 1

    #     return mask