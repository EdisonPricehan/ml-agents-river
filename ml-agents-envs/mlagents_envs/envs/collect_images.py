import os
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import matplotlib.pyplot as plt


if __name__ == '__main__':
    env_path = None
    # env_path = '/home/edison/Terrain/circular_river_collision/circular_river_collision.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_medium/circular_river_medium.x86_64'
    # env_path = '/home/edison/Terrain/circular_river_hard/circular_river_hard.x86_64'
    # env_path = '/home/edison/Terrain/riverine_training_env/riverine_training_env.x86_64'
    # env_path = '/home/edison/Research/unity-saferl-envs/medium/riverine_medium_env.x86_64'
    # env_path = '/home/edison/Research/unity-saferl-envs/easy_dr/riverine_easy_dr_env.x86_64'
    # env_path = '/home/edison/Research/unity-saferl-envs/hard_dr/riverine_hard_dr_env.x86_64'
    # env_path = None

    img_dir_name = 'sim_images/images'
    mask_dir_name = 'sim_images/masks'
    img_save_dir = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/' + img_dir_name
    mask_save_dir = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/' + mask_dir_name
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    print(f'Saving images to {img_save_dir} ...')

    max_img_num = 20
    wait_frame_num = 0

    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=1,
                                 additional_args=['-logFile', 'unity.log'])
    # allow_multiple_obs=True gives out rgb and masks observations in a list
    # encode_obs=False gives out original observation input without encoding
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False, allow_multiple_obs=True,
                            encode_obs=False, safe_rl=False, wait_frames_num=wait_frame_num)
    print(f'Env loaded!')

    i = -1
    while (i := i + 1) < max_img_num:
        obs, info = env.reset()
        # obs = env.render()
        # print(f'{obs[0].shape=} {obs[1].shape=}')
        # print(f'{len(obs)=}')
        # print(f'{obs[0]=}')

        img_path = os.path.join(img_save_dir, ('%04d' % i) + '.jpg')
        if len(obs) == 2:
            mask_path = os.path.join(mask_save_dir, ('%04d' % i) + '.png')
            plt.imsave(img_path, obs[0])
            plt.imsave(mask_path, obs[1])
        elif len(obs) == 1:
            plt.imsave(img_path, obs)
        else:
            print(f'Unknown obs size: {obs.shape}')

        # time.sleep(0.1)

    print(f'All images saved!')
    env.close()
    print(f'Env closed!')
