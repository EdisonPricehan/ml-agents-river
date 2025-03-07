import omnisafe

from mlagents_envs.base_env import CameraPose
from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.envs.env_utils import make_unity_env

from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from mlagents_envs.side_channel.agent_configuration_channel import AgentConfigurationChannel, AgentConfig
from mlagents_envs.side_channel.agent_reset_channel import AgentResetChannel
from mlagents_envs import logging_util

from omnisafe.algorithms.algo_wrapper import AlgoWrapper as Agent
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace
from omnisafe.typing import DEVICE_CPU

from typing import Any, Union, ClassVar
import numpy as np
import random
import torch
import os

import pdb
# pdb.set_trace()

import sys
print(sys.path)
# sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
# sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
# sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
# from dataset import InputChannelConfig
# from encoder.dataset import InputChannelConfig
# channel_config = InputChannelConfig.RGB_MASK

# vae_model_name = 'vae-sim-rgb_mask-16.pth'
vae_model_name = 'vae-sim-all-rgb_mask-16.pth'  # trained with image-mask from all environments

# env_path = '/home/edison/Research/unity-saferl-envs/medium/riverine_medium_env.x86_64'
env_path = '/home/edison/Research/unity-saferl-envs/medium_dr/riverine_medium_dr_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/medium_dr_ext_reset/riverine_medium_dr_ext_reset_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/easy_dr/riverine_easy_dr_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/hard_dr/riverine_hard_dr_env.x86_64'
# env_path = '/home/edison/Research/unity-saferl-envs/medium_dr_lvp/riverine_medium_dr_lvp_env.x86_64'
# env_path = None
seed = 0
env_id = 'Medium'
# env_id = 'Easy'
# env_id = 'Hard'

logger = logging_util.get_logger(__name__)
logger.setLevel(logging_util.INFO)

channel_reset = AgentResetChannel()  # TODO get this from made unity gym env


@env_register
class RiverineEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ['Easy', 'Medium', 'Hard']

    need_action_scale_wrapper = False
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = True

    _num_envs = 1
    _coordinate_observation_space: OmnisafeSpace
    _worker_id = 0  # temp workaround for duplicate unity env creation (both in env_register and real training)
    _obs_len = 16  # VAE encoded vector
    _obs_pos_len = _obs_len + 4  # pose: (x, y, z, yaw)

    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, **kwargs) -> None:
        super().__init__(env_id)
        self._device = torch.device(device)

        self.env = make_unity_env(
            env_path=env_path,
            worker_id=self._worker_id,
            seed=seed,
            max_idle_steps=50000,
        )
        self._worker_id += 1

        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space
        print(f'obs space: {self._observation_space}')
        print(f'action space: {self._action_space}')
        self._coordinate_observation_space = self._observation_space

        self.is_first_reset = True
        self.reset_pose = CameraPose()

    def get_cost_from_obs_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.env.cur_cost

    def step(self, action: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rew, cost, term, trunc, info = self.env.step(action.tolist())
        # print(f'action: {self.round_clamp_action(action).tolist()}, reward: {rew}, cost: {cost}')

        obs = obs[:self._obs_len]
        # print(f'{obs.shape=}')

        obs, rew, cost, term, trunc = (torch.as_tensor(x, dtype=torch.float32, device=self._device)
                                       for x in (obs, rew, cost, term, trunc))

        new_info = {'final_observation': obs}
        if hasattr(info['step'], 'done_reason'):
            # print(f'riverine env done reason: {info["step"].done_reason}')
            new_info['done_reason'] = info['step'].done_reason[0]
        return obs, rew, cost, term, trunc, new_info

    @staticmethod
    def round_clamp_action(float_action: torch.Tensor):
        return torch.clamp(torch.round(float_action), min=0, max=2).to(torch.int)

    def reset(self, seed: Union[int, None] = None, options: Union[dict[str, Any], None] = None)\
            -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)

        if not self.is_first_reset:
            # TODO
            # channel_reset.set_reset_pose(False, self.reset_pose.x, self.reset_pose.y, self.reset_pose.z,
            #                              self.reset_pose.yaw)
            channel_reset.set_reset_pose(True)  # use random reset for now
            print(f'Current reset pose: {self.reset_pose}')

        obs, _ = self.env.reset()
        # print(f'{obs=}')

        # assert len(obs) == self._obs_pos_len, f'Reset obs length is {len(obs)}'
        assert len(obs) == self._obs_len, f'Reset obs length is {len(obs)}'

        # if self.is_first_reset:
        #     pose = obs[self._obs_len:]
        #     print(f'First pose obs is {pose}')
        #     assert len(pose) == self._obs_pos_len - self._obs_len, f'Pose length is {len(pose)}'
        #     self.reset_pose.x = pose[0]
        #     self.reset_pose.y = pose[1]
        #     self.reset_pose.z = pose[2]
        #     self.reset_pose.yaw = pose[3]
        #     print(f'First reset pose: {self.reset_pose}')
        #     self.is_first_reset = False
        #
        # obs = obs[:self._obs_len]

        return torch.Tensor(obs), {}

    def set_seed(self, seed: int) -> None:
        logger.warning('Setting env seed is not supported!')

    def sample_action(self) -> torch.Tensor:
        return torch.as_tensor(self._action_space.sample())

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def coordinate_observation_space(self) -> OmnisafeSpace:
        return self._coordinate_observation_space


def evaluate(algo: str, env_id: str, seed_id: int, eval_time: int):
    LOG_DIR = f'./runs/{algo}-{{Medium}}/seed-{str(seed_id)}'
    global env_path, seed
    env_path = f'/home/edison/Research/unity-saferl-envs/{env_id.lower()}_dr/riverine_{env_id.lower()}_dr_env.x86_64'
    seed = seed_id

    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for item in scan_dir:
        if item.is_file() and item.name.split('.')[-1] == 'pt' and '200' in item.name:
            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name='track',
                width=128,
                height=128,
            )
            result_path = f'./eval-V3/{algo}/{env_id}/seed{seed}'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            evaluator.render(num_episodes=eval_time, save_replay_path=result_path, max_render_steps=1000)
    scan_dir.close()


if __name__ == '__main__':
    # agent = omnisafe.Agent('PPO', env_id)
    # agent = omnisafe.Agent('PPOLag', env_id)
    # agent = omnisafe.Agent('SafeLOOP', env_id)
    # agent = omnisafe.Agent('TD3', env_id)
    # agent = omnisafe.Agent('TD3Lag', env_id)
    # agent = omnisafe.Agent('SAC', env_id)
    # agent = omnisafe.Agent('PPOSaute', env_id)
    # agent = omnisafe.Agent('PCPO', env_id)
    # agent = omnisafe.Agent('PPOSimmerPID', env_id)
    # agent = omnisafe.Agent('PPOEarlyTerminated', env_id)
    # agent = omnisafe.Agent('FOCOPS', env_id)
    # agent = omnisafe.Agent('P3O', env_id)
    # agent = omnisafe.Agent('RCPO', env_id)
    # agent = omnisafe.Agent('OnCRPO', env_id)
    # agent = omnisafe.Agent('DDPGLag', env_id)
    # agent = omnisafe.Agent('SACLag', env_id)
    # agent = omnisafe.Agent('CCEPETS', env_id)
    # agent = omnisafe.Agent('CAPPETS', env_id)

    # agent = omnisafe.Agent('FOCOPS_CACD', env_id)
    agent = omnisafe.Agent('FOCOPS', env_id)

    # training
    agent.learn()
    # agent.plot(smooth=1)
    # agent.render(num_episodes=1, render_mode='rgb_array', width=128, height=128)

    # evaluation
    # for algo in ['PPO', 'PPOLag', 'FOCOPS', 'P3O', 'OnCRPO', 'DDPGLag', 'TD3Lag', 'SACLag', 'SafeLOOP', 'CCEPETS']:
    # # for algo in ['SACLag', 'SafeLOOP', 'CCEPETS']:
    # # for algo in ['PPO']:
    #     for env_id in ['Easy', 'Medium', 'Hard']:
    #     # for env_id in ['Medium']:
    #         for seed_id in range(3):
    #             evaluate(algo, env_id, seed_id, eval_time=20)




