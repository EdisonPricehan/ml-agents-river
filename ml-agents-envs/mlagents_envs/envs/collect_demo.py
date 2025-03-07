import os
import csv
from typing import Optional, List

import numpy as np

from mlagents_envs.envs.env_utils import make_unity_env
from mlagents_envs.key2action import Key2Action
import matplotlib.pyplot as plt


class DemoCollector:
    def __init__(
            self,
            env_path: str,
            demo_path: str = '',
    ):
        # Check path validity
        assert os.path.exists(env_path), f'{env_path} does not exist!'
        if not os.path.exists(demo_path):
            print(f'{demo_path} does not exist, making one...')
            os.mkdir(demo_path)

        # Define params for demo folders
        self.demo_path: str = demo_path
        self.demo_id: int = 10  # TODO
        self.demo_folder_path: Optional[str] = None
        self.demo_img_folder_path: Optional[str] = None
        self.demo_mask_folder_path: Optional[str] = None
        self.demo_csv_path: Optional[str] = None
        self.episode_step: int = 0

        # Update demo folders with current demo id
        self._update_demo_folders()

        # Init keyboard controller
        self.k2a = Key2Action()

        # Make Unity env
        self.env = make_unity_env(env_path=env_path, max_idle_steps=50000)

    def _update_demo_folders(self):
        if self.demo_path == '':
            return

        self.demo_folder_path = os.path.join(self.demo_path, f'demo{self.demo_id}')
        assert not os.path.exists(self.demo_folder_path), f'{self.demo_folder_path} already exists!'
        os.mkdir(self.demo_folder_path)

        self.demo_img_folder_path = os.path.join(self.demo_folder_path, 'images')
        self.demo_mask_folder_path = os.path.join(self.demo_folder_path, 'masks')
        self.demo_csv_path = os.path.join(self.demo_folder_path, 'traj.csv')

        os.mkdir(self.demo_img_folder_path)
        os.mkdir(self.demo_mask_folder_path)

    def _save_obs_act(self, rgb: np.ndarray, mask: np.ndarray, act: List[int]):
        if self.demo_path == '':
            return
        # TODO remove unintended no_op actions?

        img_path = os.path.join(self.demo_img_folder_path, f'{self.episode_step:04d}.jpg')
        mask_path = os.path.join(self.demo_mask_folder_path, f'{self.episode_step:04d}.png')

        plt.imsave(img_path, rgb)
        plt.imsave(mask_path, mask)

        # Write two paths and array elements as a single row
        with open(self.demo_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([img_path, mask_path] + act)

    def run(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Turn off the axes for each subplot
        for ax in axes:
            ax.axis('off')
        # Interactive plot
        plt.ion()
        # Change the save key to Shift + S to avoid conflict
        plt.rcParams['keymap.save'] = ['shift+s']

        obs = self.env.reset()
        rgb, mask, mixed = self.env.render()
        while rgb is None or mask is None or mixed is None:
            print(f'Render frame is still none, reset again...')
            obs = self.env.reset()
            rgb, mask, mixed = self.env.render()
        print(f'Render frames are ready.')

        rgb_canvas = axes[0].imshow(rgb)
        mask_canvas = axes[1].imshow(mask)
        mixed_canvas = axes[2].imshow(mixed)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        try:
            i = 0
            while i < 10000:
                # get next action either manually or randomly
                action = self.k2a.get_multi_discrete_action()  # no action if no keyboard input

                # Unintentional no_op will be not be passed to the env
                if action is None:
                    continue

                self._save_obs_act(rgb, mask, action)
                self.episode_step += 1

                obs, reward, cost, terminated, truncated, info = self.env.step(action)

                rgb, mask, mixed = self.env.render()

                rgb_canvas.set_data(rgb)
                mask_canvas.set_data(mask)
                mixed_canvas.set_data(mixed)

                plt.draw()
                plt.pause(0.001)

                if terminated or truncated:
                    self._save_obs_act(rgb, mask, action)

                    obs = self.env.reset()
                    rgb, mask, mixed = self.env.render()
                    self.episode_step = 0

                    self.demo_id += 1
                    self._update_demo_folders()
        except KeyboardInterrupt:
            self.env.close()
            plt.close()
            self.k2a.listener.join()


if __name__ == '__main__':
    # Configurable params
    env_path = '/home/edison/Research/unity-saferl-envs/medium_dr/riverine_medium_dr_env.x86_64'
    demo_path = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/demos'

    # Start demo collector
    demo_collector = DemoCollector(env_path=env_path, demo_path=demo_path)
    demo_collector.run()

