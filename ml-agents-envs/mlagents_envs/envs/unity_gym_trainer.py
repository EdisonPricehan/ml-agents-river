import time
import os
import numpy as np

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3 import PPO

import sys

sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO
import bc
from train_mirl import SaveOnBestTrainingRewardCallback
from evaluation import evaluate_policy

from env_utils import make_unity_env


# Modifiable values
env_seed = 3
train_rl = True
rl_name = "PPO"
il_name = "BC"
sample_good = False
sample_bad = False
train_il_good = False
train_il_bad = False
tmp_path = "/tmp/sb3_log/"
vae_model_name = 'vae-sim-rgb-easy.pth'
tb_log_dir = './ppo_river_tensorboard/'
tb_log_name = 'easy_ppo_compsition'
model_save_name = 'circular_easy_ppo_composition'
# env_name ="LunarLanderContinuous-v2" # 'Pendulum-v1' #"LunarLander-v2" #"CartPole-v1" #
# env_name ="unity-river"

np.set_printoptions(suppress=True)
# train_ppo_ep = 2000000
train_ppo_ep = 100000
train_il_ep = 50
train_il_bad_ep = 1
failure_steps = 5
sample_ep = 5
sample_bad_ep = 100
check_freq = int(train_ppo_ep / 10)
rng = np.random.default_rng(0)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


def train(seed: int = env_seed, use_callback: bool = True):
    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    # env_path = None
    env = make_unity_env(env_path, 1, True, seed, vae_model_name=vae_model_name)

    callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=tmp_path,
                                                verbose=1) if use_callback else None

    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=10, verbose=1,
                tensorboard_log=tb_log_dir)
    model.learn(total_timesteps=train_ppo_ep, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    # model.save('terrain_rgb_ppo_action4d')
    model.save(model_save_name)
    print(f'Model {model_save_name} has been trained and saved!')


def predict():
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
    from vae import VAE
    from dataset import InputChannelConfig
    import torch
    import matplotlib.pyplot as plt

    # plt.ion()
    # fig = plt.figure()
    # ax_rgb = fig.add_subplot(111)

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    # env_path = None
    env = make_unity_env(env_path, 1, True, seed=env_seed, vae_model_name=vae_model_name)
    print(f'Gym environment {env_path} is created!')

    # channel_config = InputChannelConfig.RGB_ONLY
    # latent_dim = 1024
    # hidden_dims = [32, 64, 128, 256, 512, 1024]
    # vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/vae-sim-rgb.pth'
    # vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    # vae_model.eval()
    # vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    # print(f'VAE model is loaded!')

    # model = PPO.load("terrain_rgb_ppo.zip")
    # model = PPO.load("terrain_rgb_ppo_action4d_bc_static.zip")
    # model = PPO.load("terrain_rgb_ppo_action4d_bc_dynamic_good.zip")
    model = PPO.load(model_save_name + '.zip')
    print(f'PPO model {model_save_name} is loaded!')

    obs = env.reset()
    episode_reward = []
    while True:
        # recon_img = vae_model.decode(torch.Tensor(obs))[0].permute((1, 2, 0)).detach().numpy() * 255
        # ax_rgb.imshow(recon_img.astype(np.uint8))
        #
        # fig.canvas.draw()
        # fig.canvas.flush_events()

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        episode_reward.append(rewards)
        # print(f'{info=}')
        if done:
            print(f'Done!')
            print(f'Episode length {len(episode_reward)}, episode reward {np.sum(episode_reward)}')
            obs = env.reset()
            episode_reward = []
            # break
        # time.sleep(0.5)
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
    # train(seed=env_seed, use_callback=train_il_good)
    predict()
    # predict_bc()
