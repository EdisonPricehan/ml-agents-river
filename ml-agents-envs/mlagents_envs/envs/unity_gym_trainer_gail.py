from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

import numpy as np
import sys
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
from train_utils import read_csv_unity


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def make_unity_env(env_directory, num_env, visual, start_index=0):
    """
    Create a wrapped, monitored Unity environment.
    """

    def make_env(rank, use_visual=True):  # pylint: disable=C0111
        def _thunk():
            unity_env = UnityEnvironment(env_directory, base_port=5000 + rank)
            env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False)
            new_logger = configure("/tmp/unity_sb3_ppo_log/", ["stdout", "csv", "tensorboard"])
            env = Monitor(env, filename=new_logger.get_dir())
            env = Monitor(env)
            return env

        return _thunk

    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])


def train():
    SEED = 42

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env = make_unity_env(env_path, 1, True)

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.00001,
        n_epochs=10,
        # seed=SEED,
    )

    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    rollouts = read_csv_unity()
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=100,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=10,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )
    gail_trainer.allow_variable_horizon = True

    # evaluate the learner before training
    # env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )

    # train the learner and evaluate again
    gail_trainer.train(100000)
    # env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )

    print("mean reward after training:", np.mean(learner_rewards_after_training))
    print("mean reward before training:", np.mean(learner_rewards_before_training))

    learner.save('terrain_rgb_ppo_action4d_gail')
    print(f'Model trained and saved!')


def predict():
    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    env = make_unity_env(env_path, 1, True)
    print(f'Gym environment created!')

    # model = PPO.load("terrain_rgb_ppo.zip")
    model = PPO.load("terrain_rgb_ppo_action1d_gail.zip")
    print(f'Model loaded!')

    obs = env.reset()
    episode_rewards = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(f'{action=} {rewards=}')
        episode_rewards.append(rewards[0])
        # print(f'{info=}')
        if done:
            print(f'Done!')
            print(f'Mean episode reward: {np.mean(episode_rewards)}')
            episode_rewards = []
            obs = env.reset()
            # break
        # env.render()


if __name__ == '__main__':
    train()
    # predict()

