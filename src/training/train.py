from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import os
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from src.envs import QueueEnv
from src.evaluation import rollout_collect_mean_q_parallel
from src.models import QueuePPOPolicy
from src.utils.arrivals import generate_arrival_params_load_new
from src.training import config as cfg

class ParallelQueueEvalCallback(BaseCallback):
    """
    Parallel evaluation using your existing rollout_collect_mean_q_parallel().
    Selects best model by LOWEST episode-average queue mean.
    """
    def __init__(
        self,
        eval_env_kwargs: dict,
        eval_freq: int,
        T: int = 4000,
        num_episodes: int = 100,
        seed: int = 123,
        n_jobs: int = None,
        best_model_save_path: str = None,
        name_prefix: str = "best_model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_kwargs = eval_env_kwargs
        self.eval_freq = int(eval_freq)
        self.T = int(T)
        self.num_episodes = int(num_episodes)
        self.seed = int(seed)
        self.n_jobs = n_jobs
        self.best_model_save_path = best_model_save_path
        self.name_prefix = name_prefix
        self.best_mean_q = np.inf
        self.total_cost = np.inf

        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.n_calls % self.eval_freq) != 0:
            return True

        # IMPORTANT:
        # rollout_collect_mean_q_parallel uses ProcessPool + fork + global model.
        # This matches your existing testing setup.
        res = rollout_collect_mean_q_parallel(
            env_kwargs=self.eval_env_kwargs,
            seed=self.seed,
            T=self.T,
            num_episodes=self.num_episodes,
            policy_fn_or_model=self.model,
            discount_factor=self.model.gamma,
            n_jobs=self.n_jobs,
        )

        mean_q = float(res["overall_mean"])
        total_cost = float(res["total_cost"])
        serve_frac = float(res["serve_fraction"])
        switch_frac = float(res["switch_fraction"])

        self.logger.record("parallel_eval/mean_queue", mean_q)
        self.logger.record("parallel_eval/total_cost", total_cost)
        self.logger.record("parallel_eval/serve_fraction", serve_frac)
        self.logger.record("parallel_eval/switch_fraction", switch_frac)

        if "mean_cap_count" in res:
            self.logger.record("parallel_eval/mean_cap_count", float(res["mean_cap_count"]))
        if "any_cap_fraction" in res:
            self.logger.record("parallel_eval/any_cap_fraction", float(res["any_cap_fraction"]))

        if self.verbose > 0:
            print(
                f"[parallel eval] steps={self.num_timesteps} | "
                f"mean_q={mean_q:.6f} | total_cost={total_cost:.6f} | "
                f"serve={serve_frac:.4f} | switch={switch_frac:.4f}"
            )

        # Save best by LOWEST total cost
        if total_cost < self.total_cost:
            self.total_cost = total_cost
            self.best_mean_q = mean_q
            self.logger.record("parallel_eval/best_mean_queue", self.best_mean_q)
            self.logger.record("parallel_eval/best_total_cost", self.total_cost)

            if self.best_model_save_path is not None:
                save_path = os.path.join(self.best_model_save_path, f"{self.name_prefix}.zip")
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"[parallel eval] new best model saved to {save_path}")

        return True

class RewardScaleWrapper(gym.RewardWrapper):
    """
    Training-only constant reward scaling.
    Does NOT change the underlying environment dynamics or infos.
    """
    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = float(scale)
        assert self.scale > 0.0

    def reward(self, reward):
        return float(reward) / self.scale

class LostArrivalPenaltyWrapper(gym.Wrapper):
    """
    Training-only penalty for arrivals censored by the queue cap.
    This is much cleaner than a barrier penalty because it activates
    only when truncation actually occurs.
    """
    def __init__(self, env, coef: float = 2.0):
        super().__init__(env)
        self.coef = float(coef)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is not None and "weighted_lost_arrivals" in info:
            reward = float(reward) - self.coef * float(info["weighted_lost_arrivals"])
        return obs, reward, terminated, truncated, info
    

# class QueueMeanAsReward(gym.Wrapper):
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)

#         # Only set the reward at the end of the episode
#         if (terminated or truncated) and info is not None and "queues_mean" in info:
#             reward = -float(info["queues_mean"])
#         else:
#             reward = 0.0

#         return obs, reward, terminated, truncated, info

def make_queue_env(rank: int,
                   base_seed: int,
                   **env_kwargs):
    """
    rank: worker id in [0, n_envs)
    base_seed: master seed for reproducibility
    env_kwargs: kwargs forwarded to QueueEnv(...)
    """
    def _init():
        env = QueueEnv(**env_kwargs)

        # training-only reward scaling
        reward_scale = float(np.max(env_kwargs["queue_cost_weights"]))
        env = RewardScaleWrapper(env, scale=reward_scale)

        # training-only cap-artifact correction
        env = LostArrivalPenaltyWrapper(env, coef=5.0)

        env = Monitor(
            env,
            info_keywords=(
                "queues_mean",
                "collisions_count",
                "stay_count",
                "switch_count",
                "served_count",
                "empty_stay_count",
                "cap_count",
                "any_cap",
                "lost_arrivals_count",
                "weighted_lost_arrivals",
            ),
        )
        # Set a distinct, reproducible seed per worker
        worker_seed = base_seed + rank
        env.reset(seed=worker_seed)  # Gymnasium seeding API
        return env
    return _init

# def make_eval_env(eval_seed: int, **env_kwargs):
#     env = QueueEnv(**env_kwargs)
#     # Wrap so EvalCallback can "maximize reward" == minimize queues_mean
#     env = QueueMeanAsReward(env)
#     # Keep Monitor for episode stats; the wrapper will replace reward but keep infos
#     env = Monitor(
#         env,
#         info_keywords=(
#             "queues_mean",
#             "collisions_count",
#             "stay_count",
#             "switch_count",
#             "served_count",
#             "empty_stay_count",
#             "cap_count",
#             "any_cap",
#         ),
#     )
#     env.reset(seed=eval_seed)
#     return env

class RolloutInfoMean(BaseCallback):
    """
    Compute per-rollout means of selected `info` keys.
    Uses *only* the samples collected in the current rollout window
    (n_envs * n_steps), then resets for the next rollout.
    """
    def __init__(self, keys=("queues_mean", "collisions_count"), log_prefix="rollout"):
        super().__init__()
        self.keys = tuple(keys)
        self.log_prefix = log_prefix
        self._reset_accums()

    def _reset_accums(self):
        # one scalar accumulator per key
        self._sum = {k: 0.0 for k in self.keys}
        self._count = 0

    def _on_rollout_start(self) -> None:
        # called once before collecting n_steps for each env
        self._reset_accums()

    def _on_step(self) -> bool:
        # called at every vectorized step; infos is a list of length n_envs
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue
            for k in self.keys:
                if k in info:
                    # cast to float to avoid dtype surprises from numpy scalars
                    self._sum[k] += float(info[k])
                    # count one sample for each key present
        # increment count by number of envs that produced an `info`
        self._count += len(infos)
        return True

    def _on_rollout_end(self) -> None:
        # called once after collecting n_steps * n_envs transitions
        if self._count == 0:
            return
        for k in self.keys:
            mean_val = self._sum[k] / self._count
            self.logger.record(f"{self.log_prefix}/{k}", mean_val)
        # (optional) also log how many samples contributed
        self.logger.record(f"{self.log_prefix}/samples", int(self._count))

def main():

    run_name = (
        f"M{cfg.M}_N{cfg.N}_u{cfg.UTILITY}"
        f"_Qmax{cfg.MAX_QUEUE_LENGTH}"
        f"_lr{cfg.LEARNING_RATE}"
        f"_nsteps{cfg.N_STEPS}"
        f"_batch{cfg.BATCH_SIZE}"
    )

    best_model_dir = cfg.CHECKPOINTS_DIR / run_name
    best_model_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = str(cfg.TB_DIR)

    arrival_params = generate_arrival_params_load_new(
        N=cfg.N,
        M=cfg.M,
        block_sum=cfg.UTILITY,
        step=cfg.ARRIVAL_STEP,
        seed=cfg.ARRIVAL_SEED,
        lam_max=cfg.LAM_MAX,
        alpha_dirichlet=cfg.ALPHA_DIRICHLET,
        enforce_nonneg=True,
    )

    env_kwargs = dict(
        M=cfg.M, N=cfg.N,
        arrival_params=arrival_params,
        queue_cost_weights=np.ones(cfg.N, dtype=np.float32),
        lambda_collision=cfg.LAMBDA_COLLISION,
        max_steps_per_run=cfg.MAX_STEPS_PER_RUN,
        max_queue_length=cfg.MAX_QUEUE_LENGTH,
    )

    vec_env = DummyVecEnv([make_queue_env(i, cfg.BASE_SEED, **env_kwargs) for i in range(cfg.N_ENVS)])

    # reward-only normalization; observations stay raw
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=cfg.GAMMA,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(cfg.BASE_SEED, using_cuda=torch.cuda.is_available())

    arrival_rates = torch.tensor(env_kwargs["arrival_params"], dtype=torch.float32)

    model = PPO(
        QueuePPOPolicy,
        vec_env,                           # vectorized env here
        learning_rate=cfg.LEARNING_RATE,
        n_steps=cfg.N_STEPS,                       # steps *per env* before each update
        batch_size=cfg.BATCH_SIZE,
        n_epochs=cfg.N_EPOCHS,                       # PPO SGD passes per update
        ent_coef=cfg.ENT_COEF,
        vf_coef=cfg.VF_COEF,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        device=device,
        clip_range=cfg.CLIP_RANGE,
        gamma=cfg.GAMMA,
        tensorboard_log=tensorboard_dir,
        policy_kwargs=dict(
            arrival_rates=arrival_rates,
            queue_cost_weights=env_kwargs["queue_cost_weights"],
            max_queue_length=env_kwargs["max_queue_length"],
        ),
        verbose=1,)

    cb_rollout_mean = RolloutInfoMean(
        keys=(
            "queues_mean",
            "collisions_count",
            "stay_count",
            "switch_count",
            "served_count",
            "empty_stay_count",
            "cap_count",
            "any_cap",
        )
    )

    # Combine your rollout-mean logger with evaluation
    callback = CallbackList([cb_rollout_mean])

    model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS, 
                tb_log_name=f"{run_name}", callback=callback)


    parallel_eval_cb = ParallelQueueEvalCallback(
        eval_env_kwargs=env_kwargs,
        eval_freq=cfg.EVAL_FREQ,         # same cadence you had before
        T=cfg.EVAL_T,                  # episode length for evaluation
        num_episodes=cfg.EVAL_EPISODES,        # your fast parallel evaluator can handle this
        seed=cfg.EVAL_SEED,
        n_jobs=None,             # or set 8 explicitly
        best_model_save_path=best_model_dir,
        name_prefix="best_model",
        verbose=1,
    )

    callback = CallbackList([cb_rollout_mean, parallel_eval_cb])

    model.learn(
        total_timesteps=cfg.ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"{run_name}_additional_steps",
        callback=callback,
    )

    # model.save(best_model_dir / "final_model.zip")
    # print(f"Saved final model to {best_model_dir / 'final_model.zip'}")

if __name__ == "__main__":
    main()