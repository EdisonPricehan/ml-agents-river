import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger

from env_utils import make_unity_env

import sys
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO

vae_model_name = 'vae-sim-rgb-easy.pth'


def train(seed: int = 1):
    expert_policy_path = 'policies/PPO_unity_river_100000_seed_3.zip'
    model_save_name = 'circular_easy_ppo_bc'
    tb_log_dir = './ppo_river_tensorboard/'
    tb_log_name = 'easy_ppo_bc'
    train_steps = 100000

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    env = make_unity_env(env_path, 1, True, seed, vae_model_name=vae_model_name)

    rng = np.random.default_rng(0)

    expert = PPO.load(expert_policy_path)

    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=400 * 50, min_episodes=100),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)
    print(f'Expert rollouts are collected!')

    logger = imit_logger.configure(tb_log_dir + tb_log_name, ["stdout", "csv", "tensorboard"])

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        demonstrations=transitions,
        batch_size=64,
        custom_logger=logger
    )

    print(f'Start training BC ...')
    bc_trainer.train(n_epochs=100, progress_bar=True)

    bc_trainer.save_policy(model_save_name)
    print(f'BC policy is saved!')

    reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print("Reward:", reward)


if __name__ == '__main__':
    train(seed=3)
