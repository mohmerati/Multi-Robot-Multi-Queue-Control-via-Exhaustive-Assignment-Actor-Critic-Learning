import matplotlib.pyplot as plt
import numpy as np

def plot_queue_evolution(greedy_res, ppo_res, T: int, title: str = ""):
    t = np.arange(T)

    plt.figure(figsize=(10, 5))
    plt.plot(t, greedy_res["time_mean"], label="Greedy")
    plt.fill_between(
        t,
        greedy_res["time_ci_lo"],
        greedy_res["time_ci_hi"],
        alpha=0.2
    )

    plt.plot(t, ppo_res["time_mean"], label="PPO")
    plt.fill_between(
        t,
        ppo_res["time_ci_lo"],
        ppo_res["time_ci_hi"],
        alpha=0.2
    )

    plt.xlabel("Time step")
    plt.ylabel("Average queue length")
    plt.title(title if title else "Average queue length over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cap_count_evolution(greedy_res, ppo_res, T: int, title: str = ""):
    t = np.arange(T)

    plt.figure(figsize=(10, 5))
    plt.plot(t, greedy_res["cap_time_mean"], label="Greedy")
    plt.fill_between(t, greedy_res["cap_time_ci_lo"], greedy_res["cap_time_ci_hi"], alpha=0.2)

    plt.plot(t, ppo_res["cap_time_mean"], label="PPO")
    plt.fill_between(t, ppo_res["cap_time_ci_lo"], ppo_res["cap_time_ci_hi"], alpha=0.2)

    plt.xlabel("Time step")
    plt.ylabel("Number of capped queues")
    plt.title(title if title else "Capped queues over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()