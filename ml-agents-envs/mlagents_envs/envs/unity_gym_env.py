import itertools
import torch
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
from gym import error, spaces

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs import logging_util

import sys
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
from vae import VAE
from dataset import InputChannelConfig


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logger = logging_util.get_logger(__name__)
GymStepResult = Tuple[np.ndarray, float, bool, Dict]


class UnityToGymWrapper(gym.Env):
    """
    Provides Gym wrapper for Unity Learning Environments.
    """

    def __init__(
            self,
            unity_env: BaseEnv,
            uint8_visual: bool = False,
            flatten_branched: bool = False,
            allow_multiple_obs: bool = False,
            action_space_seed: Optional[int] = None,
            encode_obs: bool = True,
            wait_frames_num: int = 0,
    ):
        """
        Environment initialization
        :param unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
        :param uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        :param flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than
            MultiDiscrete.
        :param allow_multiple_obs: If True, return a list of np.ndarrays as observations with the first elements
            containing the visual observations and the last element containing the array of vector observations.
            If False, returns a single np.ndarray containing either only a single visual observation or the array of
            vector observations.
        :param action_space_seed: If non-None, will be used to set the random seed on created gym.Space instances.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: Optional[DecisionSteps] = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs
        self.encode_obs = encode_obs
        self.wait_frames_num = wait_frames_num

        # Check brain configuration
        if len(self._env.behavior_specs) != 1:
            raise UnityGymException(
                "There can only be one behavior in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        self.name = list(self._env.behavior_specs.keys())[0]
        self.group_spec = self._env.behavior_specs[self.name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual
        if (
                self._get_n_vis_obs() + self._get_vec_obs_size() >= 2
                and not self._allow_multiple_obs
        ):
            logger.warning(
                "The environment contains multiple observations. "
                "You must define allow_multiple_obs=True to receive them all. "
                "Otherwise, only the first visual observation (or vector observation if"
                "there are no visual observations) will be provided in the observation."
            )

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)
        self._check_agents(len(decision_steps))
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            if self.group_spec.action_spec.discrete_size == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        elif self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )

            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        if action_space_seed is not None:
            self._action_space.seed(action_space_seed)

        # Set observations space
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        for shape in shapes:
            if uint8_visual:
                # list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))

                high = np.array([2.0] * 1024)
                list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        if self._get_vec_obs_size() > 0:
            # vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one

        # load NN models for VAE encoding and IL
        mode = 'sim'  # or 'real' or 'both'
        channel_config = InputChannelConfig.RGB_ONLY  # or 'MASK_ONLY' or 'RGB_MASK'
        vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/' + 'vae-' + \
                         mode + '-rgb' + '-random' + '.pth'  # change channel config
        assert os.path.exists(vae_model_path), f'vae model {vae_model_path} not exists!'
        latent_dim = 1024
        hidden_dims = [32, 64, 128, 256, 512, 1024]
        self.vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.vae_model.eval()
        self.vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
        print(f'VAE model {vae_model_path} is loaded!')

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()
        decision_step, _ = self._env.get_steps(self.name)
        n_agents = len(decision_step)
        self._check_agents(n_agents)
        self.game_over = False

        res: GymStepResult = self._single_step(decision_step)
        return res[0]

    def step(self, action: List[Any]) -> GymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object/list): an action provided by the environment
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        if self.game_over:
            raise UnityGymException(
                "You are calling 'step()' even though this environment has already "
                "returned done = True. You must always call 'reset()' once you "
                "receive 'done = True'."
            )
        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        action = np.array(action).reshape((1, self.action_size))

        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)
        self._env.set_actions(self.name, action_tuple)

        self._env.step()
        decision_step, terminal_step = self._env.get_steps(self.name)
        self._check_agents(max(len(decision_step), len(terminal_step)))
        if len(terminal_step) != 0:
            # The agent is done
            self.game_over = True
            return self._single_step(terminal_step)
        else:
            if self.wait_frames_num > 0:
                action = np.ones_like(action)  # doing no movements, just want the visual observation to be stable
                action_tuple.add_discrete(action)
                i = -1
                while (i := i + 1) < self.wait_frames_num:
                    self._env.set_actions(self.name, action_tuple)
                    self._env.step()
                decision_step_new, terminal_step_new = self._env.get_steps(self.name)
                assert len(terminal_step_new) == 0, 'Agent done flag should not change if no action!'
                decision_step.obs = decision_step_new.obs  # only update the obs, leave reward, info, done unchanged
            return self._single_step(decision_step)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        if self._allow_multiple_obs:
            visual_obs = self._get_vis_obs_list(info)
            visual_obs_list = []
            for obs in visual_obs:
                visual_obs_list.append(self._preprocess_single(obs[0]))
            default_observation = visual_obs_list
            if self._get_vec_obs_size() >= 1:
                default_observation.append(self._get_vector_obs(info)[0, :])
        else:
            if self._get_n_vis_obs() >= 1:
                visual_obs = self._get_vis_obs_list(info)
                default_observation = self._preprocess_single(visual_obs[0][0])
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        if self._get_n_vis_obs() >= 1:
            visual_obs = self._get_vis_obs_list(info)
            self.visual_obs = self._preprocess_single(visual_obs[0][0])

        done = isinstance(info, TerminalSteps)
        return default_observation, info.reward[0], done, {"step": info}

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        if self.uint8_visual:
            if self.encode_obs:  # obs is 1d vector of float32
                obs = torch.Tensor(single_visual_obs).permute((2, 0, 1)).unsqueeze(0)
                obs = self.vae_model.encode(obs)[0][0].detach().numpy()
            else:  # obs is 3d image of uint8
                obs = (255.0 * single_visual_obs).astype(np.uint8)
            return obs
        else:
            return single_visual_obs

    def _get_n_vis_obs(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result.append(obs_spec.shape)
        return result

    def _get_vis_obs_list(
            self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
            self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self, mode="rgb_array"):
        """
        Return the latest visual observations.
        Note that it will not render a new frame of the environment.
        """
        return self.visual_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        logger.warning("Could not seed environment %s", self.name)
        return

    @staticmethod
    def _check_agents(n_agents: int) -> None:
        if n_agents > 1:
            raise UnityGymException(
                f"There can only be one Agent in the environment but {n_agents} were detected."
            )

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param action: A scalar value representing one of the discrete actions.
        :returns: The List containing the branched actions.
        """
        return self.action_lookup[action]


if __name__ == '__main__':
    import csv
    import os
    import sys
    import io
    import torch
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    from mlagents_envs.side_channel.segmentation_receiver_channel import SegmentationReceiverChannel
    from mlagents_envs.side_channel.rgb_receiver_channel import RGBReceiverChannel
    from mlagents_envs.key2action import Key2Action

    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder")
    sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/utils")
    from vae import VAE
    from dataset import InputChannelConfig

    import gym

    import numpy as np
    # from ppo import PPO
    from stable_baselines3 import PPO, TD3
    from imitation.util import util

    import bc
    from train_utils import *

    # env_path = None  # require Unity Editor to be running
    env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/River/mlagent-ram-seg.x86_64'
    # env_path = '/home/edison/River/mlagent-ram-test2.x86_64'
    # env_path = '/home/edison/River/mlagent-ram-4D.x86_64'
    # env_path = '/home/edison/Research/ml-agents/Visual3DBall.x86_64'
    # env_path = '/home/edison/TestAgent/testball.x86_64'
    # env_path = '/home/edison/RollerBall/SlidingCube.x86_64'

    width, height = 128, 128

    channel_env = EnvironmentParametersChannel()
    channel_env.set_float_parameter("simulation_mode", 1.0)

    channel_eng = EngineConfigurationChannel()
    channel_eng.set_configuration_parameters(width=width, height=height, quality_level=1, time_scale=1,
                                             target_frame_rate=None, capture_frame_rate=None)

    channel_seg = SegmentationReceiverChannel()
    channel_rgb = RGBReceiverChannel()

    unity_env = UnityEnvironment(file_name=env_path, no_graphics=False, seed=1,
                                 side_channels=[channel_env, channel_eng, channel_seg, channel_rgb],
                                 additional_args=['-logFile', 'unity.log'])
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=False)
    obs = env.reset()
    # print(f'{env.observation_space=}')
    print(f'{env.action_space=}')
    print(f'{obs.shape=}')

    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(121)
    ax_mask = fig.add_subplot(122)

    last_mask = None
    mask_show = None
    last_obs = None
    is_mask_sync = False
    cur_sync_frame_num = 0
    min_sync_frame_num = 5
    i = 0

    save_fig = False  # whether save figures and csv
    record_bad = False  # set to False to manually record good demos

    # load NN models for VAE encoding and IL
    mode = 'sim'  # or 'real' or 'both'
    channel_config = InputChannelConfig.RGB_ONLY  # or 'MASK_ONLY' or 'RGB_MASK'
    vae_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/encoder/models/' + 'vae-' + \
                     mode + '-rgb' + '.pth'  # change channel config
    il_model_path = '/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning/weight/BC_RGB_500'
    print(f'{vae_model_path=}')
    print(f'{il_model_path=}')
    latent_dim = 1024
    hidden_dims = [32, 64, 128, 256, 512, 1024]
    vae_model = None
    # vae_model = VAE(in_channels=channel_config.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    # vae_model.eval()
    # vae_model.load_state_dict(torch.load(vae_model_path, map_location=torch.device('cpu')))
    # print(f'VAE model {vae_model_path} is loaded!')

    il_name = "BC"
    train_il_ep = 2000

    observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1024,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3])
    # action_space = gym.spaces.Discrete(9)
    datasets = ["MASK_ONLY", "RGB_ONLY", "RGB_MASK"]
    dataset = datasets[0]
    rng = np.random.default_rng(0)

    d_s = read_csv_unity()
    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=None,
        rng=rng,
        verbose=False,
    )

    # bc_policy = bc.reconstruct_policy("weight/" + il_name + "_" + dataset +"_" + str(train_il_ep))
    bc_policy = bc.reconstruct_policy(il_model_path)
    print(f'IL model is loaded!')

    trajectory_path = 'trajectories'
    trajectory_good_path = trajectory_path + '/good'
    trajectory_bad_path = trajectory_path + '/bad'
    demo_id = 3
    print(f'Current demo: {demo_id}')
    demo_name = f'demo{demo_id}'
    demo_path = os.path.join(trajectory_bad_path if record_bad else trajectory_good_path, demo_name)
    if save_fig:
        os.makedirs(demo_path, exist_ok=True)
    csv_path = os.path.join(demo_path, 'traj.csv')
    img_dir = os.path.join(demo_path, 'images')
    mask_dir = os.path.join(demo_path, 'masks')
    if save_fig:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

    obs_path = None
    mask_path = None
    waited_frames = 0
    wait_frames_num = 5

    k2a = Key2Action()  # start a new thread
    if record_bad:
        k2a.listener.stop()  # stop capturing keyboard input if using random actions
    # k2a.listener.stop()
    def update_demo_path():
        global demo_name, demo_path, csv_path, img_dir, mask_dir
        print(f'Current demo: {demo_id}')
        demo_name = f'demo{demo_id}'
        demo_path = os.path.join(trajectory_bad_path if record_bad else trajectory_good_path, demo_name)
        if save_fig:
            os.makedirs(demo_path, exist_ok=True)

        csv_path = os.path.join(demo_path, 'traj.csv')

        img_dir = os.path.join(demo_path, 'images')
        mask_dir = os.path.join(demo_path, 'masks')
        if save_fig:
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)


    actions = np.zeros([9, 4])
    actions[1, 3] = 1  # up
    actions[2, 3] = 2  # down
    actions[3, 2] = 1  # l r
    actions[4, 2] = 2  # r r
    actions[5, 1] = 1  # f
    actions[6, 1] = 2  # b
    actions[7, 0] = 1  # l
    actions[8, 0] = 2  # r

    while i < 10000:
        # get next action either manually or randomly
        action = k2a.get_multi_discrete_action()  # no action if no keyboard input
        # action = k2a.get_discrete_action()
        if record_bad and is_mask_sync:
            action = k2a.get_random_action()
            is_mask_sync = False

        # predict action using nn model
        if vae_model is not None:
            if channel_config == InputChannelConfig.MASK_ONLY:
                obs = torch.Tensor(mask_show).permute((2, 0, 1))[0].unsqueeze(0).unsqueeze(0)
            elif channel_config == InputChannelConfig.RGB_ONLY:
                obs = torch.Tensor(obs).permute((2, 0, 1)).unsqueeze(0) / 255.0
            # print(f'{obs.shape=}')
            encoding = vae_model.encode(obs)[0][0].to("cuda:0")
            acts = util.safe_to_tensor(actions).to("cuda:0")
            # obs = util.safe_to_tensor(encoding)
            # breakpoint()
            _, log_prob, entropy = bc_policy.evaluate_actions(encoding.unsqueeze(0), acts)
            print(f'{log_prob=}')
            action_il = log_prob.cpu().detach().numpy().argmax()

            action_int, _ = bc_policy.predict(encoding.cpu().detach().numpy())

            # result = 0
            # for idx, v in enumerate():
            #     if v != 0:
            #         result = 2 * idx + v
            #         break

            # if action_int > 0:
            #     idx = (action_int-1) // 2
            #     value = (action_int-1) % 2 + 1
            #     action[idx] = value
            is_mask_sync = False
            print(f'IL: {action_il=}, KEY: {action=}, PRED: {action_int=}')

        # save image-mask-action to csv only when action is not all 0
        if save_fig and obs_path is not None and mask_path is not None and any(action):
            with open(csv_path, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                # writer.writerow([obs_path, mask_path, action])
                writer.writerow([obs_path, action])
                print(f'Save img-mask-action to {csv_path}')
            i += 1
            print(f'{i=}')

        # step, if done get ready to save stuff to the new folders/file
        obs, reward, done, info = env.step(action)
        if done:
            print('Done!')
            env.reset()
            demo_id += 1
            if save_fig:
                update_demo_path()
            is_mask_sync = False
            cur_sync_frame_num = 0
            continue

        # mask = channel_seg.get_segmentation_mask()
        # rgb = channel_rgb.get_rgb()

        # if mask is None:
        #     print(f'Mask is not ready!')
        #     env.reset()
        #     mask_path = None
        #     continue

        # if mask != last_mask:
        #     mask_stream = io.BytesIO(mask)
        #     mask_show = mpimg.imread(mask_stream, format='png')
        #     # if save_fig:
        #     #     mask_path = os.path.join(mask_dir, f'{i}.png')
        #     #     plt.imsave(mask_path, mask_show)  # mask figure is over-written only when it changes
        #     last_mask = mask
        #     is_mask_sync = False
        # else:
        #     cur_sync_frame_num += 1
        #     # print(f'Synced count {cur_sync_frame_num}')
        #     if cur_sync_frame_num >= min_sync_frame_num:
        #         print(f'Mask is fully synced!')
        #         cur_sync_frame_num = 0
        #         is_mask_sync = True

        if save_fig:
            obs_path = os.path.join(img_dir, f'{i}.jpg')
            plt.imsave(obs_path, obs)  # observation figure changes frequently, so over-written every step
            mask_path = os.path.join(mask_dir, f'{i}.png')
            # plt.imsave(mask_path, mask_show)

        # rgb_stream = io.BytesIO(rgb)
        # rgb_show = mpimg.imread(rgb_stream)

        # mask_stream = io.BytesIO(mask)
        # mask_show = mpimg.imread(mask_stream, format='png')

        ax_rgb.imshow(obs)
        # ax_rgb.imshow(rgb_show)
        # ax_mask.imshow(mask_show)

        fig.canvas.draw()
        fig.canvas.flush_events()
