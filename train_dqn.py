import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from config.nodes_config import NODES_CONFIG
from env_fog import FogEnv
from env_wrapper import FogGymWrapper

def main():
    fog_env = FogEnv(
        nodes_config=NODES_CONFIG,
        episode_length=800,
        alpha=0.4, # weight for energy in reward
        beta=0.6, # weight for latency in reward
        lambda_deadline=4.0, # penalty for deadline miss
        lambda_overload=0.5, # penalty for overload
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

    # Wrap for Gym
    env = FogGymWrapper(fog_env)
    new_logger = configure("./logs_dqn/", ["stdout", "csv", "tensorboard"])

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=20000,
        learning_starts=500,
        gamma=0.99,
        tau=1.0,
        target_update_interval=500,
        train_freq=4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.10,
        verbose=1,
    )

    model.set_logger(new_logger)
    total_timesteps = 100_000
    print(f"Training DQN for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save model
    model.save("./dqn_fog_scheduler")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )

    print("\n===== Evaluation =====")
    print(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")

if __name__ == "__main__":
    main()
