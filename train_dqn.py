import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

from config.nodes_config import NODES_CONFIG
from env_fog import FogEnv
from env_wrapper import FogGymWrapper


def main():
    # =====================================================
    # 1. Create FogEnv
    # =====================================================
    fog_env = FogEnv(
        nodes_config=NODES_CONFIG,
        episode_length=300,
        alpha=0.5,
        beta=0.5,
        lambda_deadline=1.0,
        lambda_overload=0.5,
        u_max=0.9,
        e_ref=50.0,   # set rough normalization constants
        l_ref=1.0,
        seed=123,
    )

    # Wrap for Gym
    env = FogGymWrapper(fog_env)

    # =====================================================
    # 2. Configure SB3 logger
    # =====================================================
    new_logger = configure("./logs_dqn/", ["stdout", "csv", "tensorboard"])

    # =====================================================
    # 3. Create DQN model
    # =====================================================
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

    # =====================================================
    # 4. Train
    # =====================================================
    total_timesteps = 50_000
    print(f"Training DQN for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save model
    model.save("./dqn_fog_scheduler")

    # =====================================================
    # 5. Evaluate
    # =====================================================
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )

    print("\n===== Evaluation =====")
    print(f"Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")


if __name__ == "__main__":
    main()
