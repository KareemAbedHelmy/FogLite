import numpy as np
from typing import Callable, Tuple

from config.nodes_config import NODES_CONFIG
from env_fog import FogEnv

def round_robin_policy(step_idx: int, num_nodes: int, state: np.ndarray) -> int:
    """
    Very simple: cycle through nodes 0..num_nodes-1.
    """
    return step_idx % num_nodes

def least_loaded_policy(step_idx: int, num_nodes: int, state: np.ndarray) -> int:
    """
    Choose the node with the smallest (busy_until - current_time) feature.
    for each node (0 = free now, closer to 1 = busy for longer).
    """
    node_feats = state[:num_nodes]  # shape (num_nodes,)
    # smallest "time until available" -> least loaded
    return int(np.argmin(node_feats))

def run_policy(
    env: FogEnv,
    policy_fn: Callable[[int, int, np.ndarray], int],
    episodes: int = 5,
) -> Tuple[float, float, float, float]:
    """
    Run a policy for a number of episodes.
    Returns:
        avg_reward_per_step,
        avg_energy_per_task,
        avg_latency_per_task,
        deadline_miss_rate
    """
    total_reward = 0.0
    total_steps = 0

    total_energy = 0.0
    total_latency = 0.0
    total_tasks = 0

    total_deadline_misses = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        step_idx = 0

        while not done:
            action = policy_fn(step_idx, env.num_nodes, state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_steps += 1

            total_energy += info["E_i"]
            total_latency += info["L_i"]
            total_tasks += 1

            if info["deadline_miss"]:
                total_deadline_misses += 1

            state = next_state if next_state is not None else state
            step_idx += 1

    avg_reward = total_reward / max(1, total_steps)
    avg_energy = total_energy / max(1, total_tasks)
    avg_latency = total_latency / max(1, total_tasks)
    miss_rate = total_deadline_misses / max(1, total_tasks)

    return avg_reward, avg_energy, avg_latency, miss_rate

def main():
    env = FogEnv(
        nodes_config=NODES_CONFIG,
        episode_length=500,
        alpha=0.4, # weight for energy in reward
        beta=0.6, # weight for latency in reward
        lambda_deadline=4.0, # penalty for deadline miss
        lambda_overload=0.5, # penalty for overload
        u_max=0.9,
        e_ref=20.0,
        l_ref=2.0,
        task_length_range=(50.0, 250.0),
        deadline_slack_range=(1.0, 3.0),
        interarrival_range=(0.0, 0.05),
        seed=123,
    )

    print("Running Round Robin...")
    rr_stats = run_policy(env, round_robin_policy, episodes=5)
    print(f"  Avg reward per step:   {rr_stats[0]:.4f}")
    print(f"  Avg energy per task:   {rr_stats[1]:.4f} J")
    print(f"  Avg latency per task:  {rr_stats[2]:.4f} s")
    print(f"  Deadline miss rate:    {rr_stats[3]*100:.2f}%")

    print("\nRunning Least Loaded...")
    ll_stats = run_policy(env, least_loaded_policy, episodes=5)
    print(f"  Avg reward per step:   {ll_stats[0]:.4f}")
    print(f"  Avg energy per task:   {ll_stats[1]:.4f} J")
    print(f"  Avg latency per task:  {ll_stats[2]:.4f} s")
    print(f"  Deadline miss rate:    {ll_stats[3]*100:.2f}%")

if __name__ == "__main__":
    main()