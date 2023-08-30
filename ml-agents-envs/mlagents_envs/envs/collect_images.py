import os
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

import matplotlib.pyplot as plt


if __name__ == '__main__':
    env_path = '/home/edison/Terrain/circular_river_4d/circular_river_action4d.x86_64'
    img_dir_name = 'sim_images'
    img_save_dir = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/' + img_dir_name
    os.makedirs(img_save_dir, exist_ok=True)
    print(f'Saving images to {img_save_dir} ...')

    max_img_num = 2000
    wait_frame_num = 5

    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=1,
                                 additional_args=['-logFile', 'unity.log'])
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False)
    print(f'Env loaded!')

    i = -1
    while (i := i + 1) < max_img_num:
        obs = env.reset()
        j = -1
        while (j := j + 1) < wait_frame_num:
            obs, _, _, _ = env.step([1] * 4)
        # print(f'{obs.shape=}')
        obs_path = os.path.join(img_save_dir, ('%04d' % i) + '.jpg')
        plt.imsave(obs_path, obs)
        time.sleep(0.1)

    print(f'All images saved!')
