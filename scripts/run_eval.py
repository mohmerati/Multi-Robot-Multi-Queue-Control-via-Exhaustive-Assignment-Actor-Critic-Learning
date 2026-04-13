from src.training import config as cfg
import numpy as np
from src.evaluation.eval import compare_policies_mean_q
from stable_baselines3 import PPO
from src.utils.arrivals import generate_arrival_params_load_new
import numpy as np
from src.evaluation.plotting import plot_queue_evolution, plot_cap_count_evolution

def main():
    # -------------------------
    # Rebuild the exact scenario
    # -------------------------
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
        M=cfg.M,
        N=cfg.N,
        arrival_params=arrival_params,
        queue_cost_weights=np.ones(cfg.N, dtype=np.float32),
        lambda_collision=cfg.LAMBDA_COLLISION,
        max_steps_per_run=cfg.MAX_STEPS_PER_RUN,
        max_queue_length=cfg.MAX_QUEUE_LENGTH,
    )

    # -------------------------
    # Rebuild training run name
    # -------------------------
    run_name = (
        f"M{cfg.M}_N{cfg.N}_u{cfg.UTILITY}"
        f"_Qmax{cfg.MAX_QUEUE_LENGTH}"
        f"_lr{cfg.LEARNING_RATE}"
        f"_nsteps{cfg.N_STEPS}"
        f"_batch{cfg.BATCH_SIZE}"
    )

    best_model_dir = cfg.CHECKPOINTS_DIR / run_name
    best_model_path = best_model_dir / "best_model.zip"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint at {best_model_path}. "
            "Make sure training finished and the evaluation callback saved the best model."
        )

    # -------------------------
    # Load trained PPO model
    # -------------------------
    model = PPO.load(str(best_model_path))

    # -------------------------
    # Evaluation scenario
    # -------------------------
    eval_env_kwargs = env_kwargs.copy()
    # If you want to override anything for evaluation, do it here:
    # eval_env_kwargs["max_steps_per_run"] = cfg.EVAL_T
    # eval_env_kwargs["arrival_params"] = np.random.permutation(eval_env_kwargs["arrival_params"])

    greedy_res, ppo_res = compare_policies_mean_q(
        env_kwargs=eval_env_kwargs,
        seed=cfg.EVAL_SEED,
        T=cfg.EVAL_T,
        num_episodes=cfg.EVAL_EPISODES,
        model=model,
    )

    # -------------------------
    # Output directory for figures
    # -------------------------
    fig_dir = cfg.OUTPUTS_DIR / "figures" / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Plot and save figures
    # -------------------------
    plot_queue_evolution(
        greedy_res,
        ppo_res,
        T=cfg.EVAL_T,
        title=f"Queue evolution: M={eval_env_kwargs['M']}, N={eval_env_kwargs['N']}"
    )
    import matplotlib.pyplot as plt
    plt.savefig(fig_dir / "queue_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_cap_count_evolution(
        greedy_res,
        ppo_res,
        T=cfg.EVAL_T,
        title=f"Capped queues: M={eval_env_kwargs['M']}, N={eval_env_kwargs['N']}"
    )
    plt.savefig(fig_dir / "cap_count_evolution.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figures to: {fig_dir}")
    print(f"Loaded model from: {best_model_path}")

if __name__ == "__main__":
    main()