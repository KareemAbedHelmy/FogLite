import numpy as np
from typing import Callable, Tuple
from stable_baselines3 import DQN
from config.nodes_config import NODES_CONFIG
from env_fog import FogEnv
from env_wrapper import FogGymWrapper
from baselines import round_robin_policy, least_loaded_policy

def random_policy(step_idx: int, num_nodes: int, state: np.ndarray) -> int:
    """Uniform random choice of node."""
    return np.random.randint(0, num_nodes)

def run_policy_env(
    fog_env: FogEnv,
    policy_fn: Callable[[int, int, np.ndarray], int],
    episodes: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Run a (non-RL) policy directly on FogEnv.

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
    total_job_latency = 0.0
    total_job_energy = 0.0
    total_job_misses = 0
    total_jobs = 0

    for ep in range(episodes):
        state = fog_env.reset()
        done = False
        step_idx = 0

        while not done:
            action = policy_fn(step_idx, fog_env.num_nodes, state)
            next_state, reward, done, info = fog_env.step(action)

            total_reward += reward
            total_steps += 1

            total_energy += info["E_i"]
            total_latency += info["L_i"]
            total_tasks += 1
            if info["deadline_miss"]:
                total_deadline_misses += 1

            state = next_state if next_state is not None else state
            step_idx += 1
        # After episode ends, gather job stats if applicable
        for job in fog_env.completed_jobs:
            total_job_latency += job["latency"]
            total_job_energy += job["energy"]
            total_jobs += 1
            total_job_misses += 1 if job["deadline_miss"] else 0   
            
             
    avg_reward = total_reward / max(1, total_steps)
    avg_energy = total_energy / max(1, total_tasks)
    avg_latency = total_latency / max(1, total_tasks)
    miss_rate = total_deadline_misses / max(1, total_tasks)
    if total_jobs > 0:
        avg_job_latency = total_job_latency / total_jobs
        avg_job_energy = total_job_energy / total_jobs
        job_miss_rate = total_job_misses / total_jobs
    else:
        avg_job_latency = 0.0
        avg_job_energy = 0.0
        job_miss_rate = 0.0
    return avg_reward, avg_energy, avg_latency, miss_rate, avg_job_latency, avg_job_energy, job_miss_rate

def run_dqn_policy(
    model_path: str,
    fog_env: FogEnv,
    episodes: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the trained DQN model on FogEnv using the same metrics.
    """
    wrapped_env = FogGymWrapper(fog_env)
    model = DQN.load(model_path, env=wrapped_env)

    total_reward = 0.0
    total_steps = 0
    total_energy = 0.0
    total_latency = 0.0
    total_tasks = 0
    total_deadline_misses = 0
    total_job_latency = 0.0
    total_job_energy = 0.0
    total_job_misses = 0
    total_jobs = 0

    for ep in range(episodes):
        obs, _ = wrapped_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = wrapped_env.step(action)

            total_reward += reward
            total_steps += 1

            total_energy += info["E_i"]
            total_latency += info["L_i"]
            total_tasks += 1

            if info["deadline_miss"]:
                total_deadline_misses += 1

            obs = next_obs
    for job in fog_env.completed_jobs:
        total_job_latency += job["latency"]
        total_job_energy += job["energy"]
        total_jobs += 1
        total_job_misses += 1 if job["deadline_miss"] else 0

    avg_reward = total_reward / max(1, total_steps)
    avg_energy = total_energy / max(1, total_tasks)
    avg_latency = total_latency / max(1, total_tasks)
    miss_rate = total_deadline_misses / max(1, total_tasks)
    if total_jobs > 0:
        avg_job_latency = total_job_latency / total_jobs
        avg_job_energy = total_job_energy / total_jobs
        job_miss_rate = total_job_misses / total_jobs
    else:
        avg_job_latency = 0.0
        avg_job_energy = 0.0
        job_miss_rate = 0.0

    return avg_reward, avg_energy, avg_latency, miss_rate, avg_job_latency, avg_job_energy, job_miss_rate

def print_results(name: str, stats: Tuple[float, float, float, float]):
    avg_reward, avg_energy, avg_latency, miss_rate,avg_job_latency,avg_job_energy,job_miss_rate = stats
    print(f"=== {name} ===")
    print(f"  Avg reward per step:   {avg_reward:.4f}")
    print(f"  Avg energy per task:   {avg_energy:.4f} J")
    print(f"  Avg latency per task:  {avg_latency:.4f} s")
    print(f"  Deadline miss rate:    {miss_rate*100:.2f}%")
    print(f"  Avg job latency:       {avg_job_latency:.4f} s")
    print(f"  Avg job energy:        {avg_job_energy:.4f} J")
    print(f"  Job deadline miss rate:{job_miss_rate*100:.2f}%")
    print()

def main():
    # Create a fresh FogEnv for evaluation
    fog_env = FogEnv(
        nodes_config=NODES_CONFIG,
        episode_length=800,
        alpha=0.4,
        beta=0.6,
        lambda_deadline=4.0,
        lambda_overload=0.5,
        u_max=0.9,
        e_ref=20.0,
        l_ref=2.0,
        task_length_range=(1000.0, 10000.0),
        deadline_slack_range=(1.0, 5.0),
        interarrival_range=(0.05, 0.2),
        use_task_dependence=True,
        max_stages_per_job= 3,
        handoff_latency=0.02,
        input_size_range=(0.1, 0.8),  # in MB
        output_size_range=(0.02, 0.25),  # in MB
        handoff_bandwidth_mbps= 50.0,  # between fog nodes
        seed=123,
    )
    episodes = 10
    # Heuristic policies
    rr_stats = run_policy_env(fog_env, round_robin_policy, episodes=episodes)
    ll_stats = run_policy_env(fog_env, least_loaded_policy, episodes=episodes)
    rand_stats = run_policy_env(fog_env, random_policy, episodes=episodes)
    # DQN policy
    dqn_stats = run_dqn_policy("./dqn_fog_scheduler.zip", fog_env, episodes=episodes)
    print_results("Round Robin", rr_stats)
    print_results("Least Loaded", ll_stats)
    print_results("Random", rand_stats)
    print_results("DQN (learned)", dqn_stats)

if __name__ == "__main__":
    main()
