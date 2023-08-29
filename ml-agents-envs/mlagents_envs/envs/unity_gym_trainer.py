import time

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3 import PPO
import sys

sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO
import bc
from train_mirl import SaveOnBestTrainingRewardCallback
from evaluation import evaluate_policy

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import os
import numpy as np

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
            width, height = 128, 128

            channel_env = EnvironmentParametersChannel()
            channel_env.set_float_parameter("simulation_mode", 1.0)

            channel_eng = EngineConfigurationChannel()
            channel_eng.set_configuration_parameters(width=width, height=height, quality_level=1, time_scale=1,
                                                     target_frame_rate=None, capture_frame_rate=None)

            unity_env = UnityEnvironment(env_directory,
                                         # base_port=5000 + rank,
                                         no_graphics=False,
                                         seed=1,
                                         side_channels=[channel_env, channel_eng],
                                         additional_args=['-logFile', 'unity.log'])
            env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False)
            # new_logger = configure("/tmp/unity_sb3_ppo_log/", ["stdout", "csv", "tensorboard"])
            # env = Monitor(env, filename=new_logger.get_dir())
            # env = Monitor(env)
            return env

        return _thunk

    if visual:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        return DummyVecEnv([make_env(rank, use_visual=False)])


train_rl = True
rl_name = "PPO"
il_name = "BC"
sample_good = True
sample_bad = False
train_il_good = True
train_il_bad = False
tmp_path = "/tmp/sb3_log/"
# env_name ="LunarLanderContinuous-v2" # 'Pendulum-v1' #"LunarLander-v2" #"CartPole-v1" #
env_name ="unity-river"

np.set_printoptions(suppress=True)
# train_ppo_ep = 2000000
train_ppo_ep = 100000
train_il_ep = 50
train_il_bad_ep = 1
failure_steps = 5
sample_ep = 5
sample_bad_ep = 100
check_freq = int(train_ppo_ep/10)
rng = np.random.default_rng(0)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def train(use_callback: bool = True):
    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    # env_path = None
    env = make_unity_env(env_path, 1, True)

    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=tmp_path, verbose=1) if use_callback else None

    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=10, verbose=1,
                tensorboard_log="./ppo_river_tensorboard/")
    model.learn(total_timesteps=100000, progress_bar=True, tb_log_name="PPO_BC_DYNAMIC_GOOD",
                callback=callback)
    # model.save('terrain_rgb_ppo_action4d')
    model.save('terrain_rgb_ppo_action4d_bc_dynamic_good')
    print(f'Model trained and saved!')


def predict():
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
    from vae import VAE
    from dataset import InputChannelConfig
    import torch
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(111)

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env = make_unity_env(env_path, 1, True)
    print(f'Gym environment created!')

    channel_config = InputChannelConfig.RGB_ONLY
    latent_dim = 1024
    hidden_dims = [32, 64, 128, 256, 512, 1024]
    vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-rgb.pth'
    vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    print(f'VAE model is loaded!')

    # model = PPO.load("terrain_rgb_ppo.zip")
    # model = PPO.load("terrain_rgb_ppo_action4d_bc_static.zip")
    model = PPO.load("terrain_rgb_ppo_action4d_bc_dynamic_good.zip")
    print(f'PPO model is loaded!')

    obs = env.reset()
    episode_reward = []
    while True:
        recon_img = vae_model.decode(torch.Tensor(obs))[0].permute((1, 2, 0)).detach().numpy() * 255
        ax_rgb.imshow(recon_img.astype(np.uint8))

        fig.canvas.draw()
        fig.canvas.flush_events()

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        episode_reward.append(rewards)
        print(f'{action=}, step {rewards=}, episode reward {np.sum(episode_reward)}')
        # print(f'{info=}')
        if done:
            print(f'Done!')
            obs = env.reset()
            break
        time.sleep(0.5)
        # env.render()


def predict_bc():
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
    from vae import VAE
    from dataset import InputChannelConfig
    import torch
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(111)

    env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    # env_path = None
    env = make_unity_env(env_path, 1, True)
    print(f'Gym environment created!')

    channel_config = InputChannelConfig.RGB_ONLY
    latent_dim = 1024
    hidden_dims = [32, 64, 128, 256, 512, 1024]
    vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-rgb.pth'
    vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    print(f'VAE model is loaded!')

    bc_policy = bc.reconstruct_policy('/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/weight/BC_RGB_1000')
    print(f'bc policy is loaded!')

    print(f'Start predicting ...')
    obs = env.reset()
    episode_reward = []
    while True:
        recon_img = vae_model.decode(torch.Tensor(obs))[0].permute((1, 2, 0)).detach().numpy() * 255
        ax_rgb.imshow(recon_img.astype(np.uint8))

        fig.canvas.draw()
        fig.canvas.flush_events()

        action, _ = bc_policy.predict(obs)

        obs, reward, done, info = env.step(action)

        episode_reward.append(reward)
        print(f'{action=}, {reward=}, episode_reward {np.sum(episode_reward)}')
        # print(f'{info=}')
        # time.sleep(2)
        if done:
            print(f'Done')
            break
            obs = env.reset()


if __name__ == '__main__':
    # train(use_callback=True)
    predict()
    # predict_bc()
