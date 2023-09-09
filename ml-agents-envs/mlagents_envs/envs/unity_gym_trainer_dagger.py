import tempfile

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util import logger as imit_logger

from env_utils import make_unity_env

import sys

sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO

vae_model_name = 'vae-sim-rgb-easy.pth'


def train(seed: int = 0):
    expert_policy_path = 'policies/PPO_unity_river_100000_seed_3.zip'
    model_save_name = 'circular_easy_ppo_dagger'
    tb_log_dir = './ppo_river_tensorboard/'
    tb_log_name = 'easy_ppo_dagger_1'
    train_steps = 100000

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    env = make_unity_env(env_path, 1, True, seed, vae_model_name=vae_model_name)

    rng = np.random.default_rng(seed)

    expert = PPO.load(expert_policy_path)

    logger = imit_logger.configure(tb_log_dir + tb_log_name, ["stdout", "csv", "tensorboard"])

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=64,
        custom_logger=logger,
        rng=rng,
    )

    with tempfile.TemporaryDirectory(prefix="dagger_river_easy_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            custom_logger=logger,
            rng=rng,
        )
        dagger_trainer.allow_variable_horizon = True
        dagger_trainer.train(train_steps, rollout_round_min_episodes=3, rollout_round_min_timesteps=512,
                             bc_train_kwargs={'n_epochs': 1,
                                              'log_interval': 10,
                                              'log_rollouts_n_episodes': 10})
        dagger_trainer.save_policy(model_save_name)
        print(f'Dagger policy is saved as {model_save_name}')

    reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
    print("Reward:", reward)


if __name__ == '__main__':
    train(seed=3)

