import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import IntEnum
from collections import defaultdict

import gymnasium as gym
from gymnasium import error, spaces

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.base_env import CameraPose
from mlagents_envs import logging_util
from mlagents_envs.envs.action_flattener import DroneActionFlattener


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logger = logging_util.get_logger(__name__)

# For Gym: (obs, reward, done, info)
GymStepResult = Tuple[np.ndarray, float, bool, Dict]
# For Gymnasium: (obs, reward, terminated, truncated, info)
GymnasiumStepResult = Tuple[np.ndarray, float, bool, bool, Dict]
# For omnisafe: (obs, reward, cost, terminated, truncated, info)
GymSafeStepResult = Tuple[np.ndarray, float, float, bool, bool, Dict]


# Granular feedback of the done reason before episode reset
class DoneReason(IntEnum):
    # critical failure reasons
    Collision = 0
    OutOfVolumeHorizontal = 1
    OutOfVolumeVertical = 2
    # loose failure reasons
    YawOverDeviation = 3
    Idle = 4
    MaxStepReached = 5
    # success reason
    Success = 6


# Cost constants
COST_LOOSE: float = 0.5  # For less severe terminal conditions, dr = [3, 4, 5]
COST_TIGHT: float = 1.0  # For severe terminal conditions, dr = [0, 1, 2]


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
            safe_rl: bool = False,
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
        :param safe_rl: whether include 'cost' in env step result
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.rgb_obs = None  # 3-channel rgb
        self.mask_obs = None  # 4-channel mask (png)
        self.mixed_obs = None  # 4-channel rgb+mask

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: Optional[DecisionSteps] = None
        self._flattener = None

        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False
        self._allow_multiple_obs = allow_multiple_obs

        # Whether to return "cost" in step result of CMDP
        self.safe_rl = safe_rl

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
                    # self._flattener = ActionFlattener(branches)
                    self._flattener = DroneActionFlattener(branches)
                    self._action_space = self._flattener.action_space  # Discrete action space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)
        elif self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            # By default, the continuous action value is in [-1, 1]
            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        # Seed action space seed
        if action_space_seed is not None:
            self._action_space.seed(action_space_seed)

        # Set observations spaces
        list_spaces: List[gym.Space] = []
        shapes = self._get_vis_obs_shape()
        print(f'Unity gym visual observation shape: {shapes}')

        # Store visual observation first
        for shape in shapes:
            if self.uint8_visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))

        # Store vector observation last
        if self._get_vec_obs_size() > 0:
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))

        if self._allow_multiple_obs:
            self._observation_space = spaces.Tuple(list_spaces)
        else:
            self._observation_space = list_spaces[0]  # only return the first one, 3 channel RGB by default

        print(f'Unity gym observation space: {self.observation_space=}')
        print(f'Unity gym action space: {self.action_space=}')

        # Statistical info
        self.ep_rew = 0
        self.ep_len = 0
        self.done_reason_stat = defaultdict(lambda: 0)

    def reset(self, seed=None) -> Union[List[np.ndarray], np.ndarray]:
        """
        Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """
        self._env.reset()

        decision_step, _ = self._env.get_steps(self.name)
        n_agents = len(decision_step)
        self._check_agents(n_agents)
        self.game_over = False

        self.ep_rew = 0
        self.ep_len = 0

        res: Union[GymStepResult, GymSafeStepResult] = self._single_step(decision_step)
        return res[0]

    def step(self, action: List[Any]) -> Union[GymStepResult, GymnasiumStepResult, GymSafeStepResult]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a:
        Gym tuple (observation, reward, done, info) or
        Gymnasium tuple (observation, reward, terminated, truncated, info) or
        Omnisafe CMDP tuple (observation, reward, cost, terminated, truncated, info)
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

        # Action clipping
        act = np.array(action).reshape((1, self.action_size))  # (Number of agents X action size)
        action = np.clip(act, 0, 2)  # Only for multi-discrete drone action
        # print(f'continuous clipped action: {action}')

        # Set action tuple
        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)

        # Execute action
        self._env.set_actions(self.name, action_tuple)
        self._env.step()

        # Get results after the executed action
        decision_step, terminal_step = self._env.get_steps(self.name)
        self._check_agents(max(len(decision_step), len(terminal_step)))

        if len(terminal_step) != 0:
            # The agent is done
            self.game_over = True
            return self._single_step(terminal_step)
        else:
            return self._single_step(decision_step)

    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) \
            -> Union[GymStepResult, GymnasiumStepResult, GymSafeStepResult]:
        # Collect and preprocess visual observations
        if self._allow_multiple_obs:
            # Each visual obs is 4-dimensional: (agent count, w/h, h/w, channel)
            visual_obs_list = self._get_vis_obs_list(info)
            visual_obs_list_new = []
            for obs in visual_obs_list:
                visual_obs_list_new.append(self._preprocess_single(obs[0]))
            default_observation = visual_obs_list_new  # [(w/h, h/w, channel)]

            # Save rgb and mask for rendering
            if len(visual_obs_list_new) == 2:
                self.rgb_obs, self.mask_obs = default_observation[0], default_observation[1]
                self.mixed_obs = self._preprocess_double_obs(self.rgb_obs, self.mask_obs)
        else:
            if self._get_n_vis_obs() >= 1:
                visual_obs = self._get_vis_obs_list(info)[0][0]
                default_observation = self._preprocess_single(visual_obs)  # (w/h, h/w, channel)
            else:
                default_observation = self._get_vector_obs(info)[0, :]

        terminated = isinstance(info, TerminalSteps) and info.done_reason != DoneReason.MaxStepReached
        truncated = isinstance(info, TerminalSteps) and info.done_reason == DoneReason.MaxStepReached

        # Update statistics
        self.ep_rew += info.reward[0]
        self.ep_len += 1

        # Update reward and cost
        cur_reward = info.reward[0]
        if abs(cur_reward) > 1e-6:
            cur_reward = 1  # Make immediate reward of a meaningful step to be 1
        cur_cost = 0

        if self.safe_rl:
            if terminated or truncated:  # Terminal rewards and costs
                # Update done stats
                # print(f'{info.done_reason=}')
                self.done_reason_stat[info.done_reason[0]] += 1

                # update reward and cost (exclusively)
                if (info.done_reason[0] == DoneReason.YawOverDeviation.value or
                        info.done_reason[0] == DoneReason.Idle.value or
                        info.done_reason[0] == DoneReason.MaxStepReached.value):  # loose constraints
                    cur_reward = 0
                    cur_cost = COST_LOOSE
                elif info.done_reason == DoneReason.Success:  # success
                    pass
                else:  # tight constraints
                    cur_reward = 0
                    cur_cost = COST_TIGHT
            else:  # Immediate rewards and costs
                # cur_cost = self.get_water_percentage_cost(self.mask_obs)
                # cur_cost = self.get_water_iou_cost(self.mask_obs)
                pass

        # Compose step results
        if self.safe_rl:
            # CMDP step results
            return default_observation, cur_reward, cur_cost, terminated, truncated, {'step': info}
        else:
            # Gymnasium step results
            if terminated:
                # print(f'Episode reward: {self.ep_rew}, episode length: {self.ep_len}')
                return (default_observation, cur_reward, terminated, truncated,
                        {"step": info, "episode": {'r': self.ep_rew, 'l': self.ep_len}})
            else:
                return default_observation, cur_reward, terminated, truncated, {}

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        """
        Scale observation to correct range and type if necessary
        :param single_visual_obs:
        :return:
        """
        if self.uint8_visual:
            obs = (255.0 * single_visual_obs).astype(np.uint8)
            return obs
        else:
            return single_visual_obs

    def _preprocess_double_obs(self, rgb_obs: np.ndarray, mask_obs: np.ndarray) -> np.ndarray:
        """
        Concat the rgb and mask observations (mask as the alpha channel)
        :param rgb_obs: H x W x C=3 jpg image
        :param mask_obs: H x W x C=4 png rgba image
        :return: H x W x C=4 uint8 image
        """
        assert self.uint8_visual, f'Need to enable uint8_visual'
        assert rgb_obs.shape[-1] == 3, f'rgb obs has wrong shape {rgb_obs.shape}'
        assert mask_obs.shape[-1] == 4, f'mask obs has wrong shape {mask_obs.shape}'

        # print(f'rgb shape: {rgb_obs.shape}')
        # print(f'mask shape: {mask_obs.shape}')

        rgb_mask = np.concatenate([rgb_obs, mask_obs[..., 0][..., np.newaxis]], axis=2)  # H x W x C=4
        return rgb_mask

    def get_water_percentage_cost(self, mask_obs: np.ndarray) -> float:
        # TODO clearly define this cost
        assert mask_obs is not None, f'mask obs is None'
        assert len(mask_obs.shape) == 3, f'mask obs has wrong dimension {mask_obs.shape}'
        assert mask_obs.shape[-1] == 4, f'mask obs has wrong channel size {mask_obs.shape}'

        h, w, c = mask_obs.shape
        if mask_obs.dtype is not np.uint8:
            mask_int = (255 * mask_obs).astype(np.uint8)
        else:
            mask_int = mask_obs

        num_all_pixels = h * w
        num_water_pixels = np.sum([1 if e > 0 else 0 for e in mask_int[..., 0].flatten()])
        water_ratio = 1.0 * num_water_pixels / num_all_pixels
        # print(f'all pixels: {num_all_pixels}, water pixels: {num_water_pixels}, ratio: {water_ratio}')
        return 1 if water_ratio < self.min_water_ratio_thr else 0

    def get_water_iou_cost(self, mask_obs: np.ndarray) -> float:
        """
        Calculate the IoU-based cost for the water mask observation against a predefined trapezoidal mask.

        Args:
            mask_obs (np.ndarray): Observation mask (H x W x C) with 4 channels.

        Returns:
            float: IoU-based cost, calculated as 1 - IoU.
        """
        assert mask_obs is not None, f'mask obs is None'
        assert len(mask_obs.shape) == 3, f'mask obs has wrong dimension {mask_obs.shape}'
        assert mask_obs.shape[-1] == 4, f'mask obs has wrong channel size {mask_obs.shape}'

        h, w, c = mask_obs.shape
        assert h == w, f'Expect square observation, given ({h}, {w}).'
        if mask_obs.dtype is not np.uint8:
            mask_int = (255 * mask_obs).astype(np.uint8)
        else:
            mask_int = mask_obs

        mask_int = mask_int[..., 0]  # Only need single channel
        mask_trapezoid = self.create_trapezoidal_mask(
            side_length=h,
            top_width=4,
            down_width=10,
            trapezoid_height=13,
        )  # TODO these values can be percentages

        # Compute intersection and union directly
        intersection = np.logical_and(mask_int == 255, mask_trapezoid == 255).sum()
        union = np.logical_or(mask_int == 255, mask_trapezoid == 255).sum()

        # Avoid division by zero
        iou = intersection / (union + 1e-6)

        # Calculate IoU-based cost
        iou_cost = 1.0 - iou
        return iou_cost

    def create_trapezoidal_mask(
        self,
        side_length: int,
        top_width: int,
        down_width: int,
        trapezoid_height: int,
    ) -> np.ndarray:
        """
        Create a trapezoidal mask for a square observation, with the trapezoid's bottom side aligned to the bottom of the observation.

        Args:
            side_length (int): Side length of the square observation (height and width of the mask).
            top_width (int): Width of the trapezoid at the top.
            down_width (int): Width of the trapezoid at the bottom.
            trapezoid_height (int): Height of the trapezoid.

        Returns:
            np.ndarray: A binary mask (2D array) with the trapezoidal region filled with 1s.
        """
        assert top_width < side_length
        assert down_width < side_length
        assert trapezoid_height < side_length
        assert top_width <= down_width

        # Initialize the mask with zeros
        mask = np.zeros((side_length, side_length), dtype=np.uint8)

        # Calculate the vertical positions for the trapezoid
        bottom_y = side_length  # Bottom edge of the observation
        top_y = bottom_y - trapezoid_height  # Top edge of the trapezoid

        # Calculate the horizontal positions for the top and bottom edges of the trapezoid
        top_left = (side_length - top_width) // 2
        top_right = top_left + top_width
        bottom_left = (side_length - down_width) // 2
        bottom_right = bottom_left + down_width

        # Fill in the trapezoidal area
        for y in range(top_y, bottom_y):
            # Interpolate the width of the trapezoid at the current height
            alpha = (y - top_y) / trapezoid_height
            current_left = int((1 - alpha) * top_left + alpha * bottom_left)
            current_right = int((1 - alpha) * top_right + alpha * bottom_right)

            # Fill the row in the trapezoidal range
            mask[y, current_left:current_right] = 255

        return mask

    def _get_n_vis_obs(self) -> int:
        """
        Get the number of visual observations
        :return:
        """
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> List[Tuple]:
        """
        Get all shapes of visual observations
        :return:
        """
        result: List[Tuple] = []
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                result.append(obs_spec.shape)
        return result

    def _get_vis_obs_list(self, step_result: Union[DecisionSteps, TerminalSteps]
                          ) -> List[np.ndarray]:
        """
        Get the data of all visual observations
        :param step_result:
        :return:
        """
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
            self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> np.ndarray:
        """
        Get the data of all vector observations
        :param step_result:
        :return:
        """
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        """
        Get all shapes of the vector observations
        :return:
        """
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self, mode="rgb_array") -> Tuple[Optional[np.ndarray], ...]:
        """
        Return the latest visual observations.
        Note that it will not render a new frame of the environment.
        """
        return self.rgb_obs, self.mask_obs, self.mixed_obs

    def close(self) -> None:
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        print('Done reason statistics: ')
        for reason, count in self.done_reason_stat.items():
            print(f'{DoneReason(reason).name}: {count}')

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
    def observation_space(self) -> gym.Space:
        return self._observation_space


if __name__ == '__main__':
    # Play with safe riverine environment
    from mlagents_envs.envs.env_utils import make_unity_env
    from mlagents_envs.key2action import Key2Action
    import matplotlib.pyplot as plt

    # Params
    env_path = '/home/edison/Research/unity-saferl-envs/medium_dr/riverine_medium_dr_env.x86_64'

    env = make_unity_env(env_path=env_path, max_idle_steps=50000)
    obs = env.reset()
    rgb, mask, mixed = env.render()

    k2a = Key2Action()  # Start a new thread

    fig, ax = plt.subplots(1, 3)
    rgb_canvas = ax[0].imshow(rgb)
    mask_canvas = ax[1].imshow(mask)
    mixed_canvas = ax[2].imshow(mixed)

    try:
        i = 0
        while i < 10000:
            # get next action either manually or randomly
            action = k2a.get_multi_discrete_action()  # no action if no keyboard input

            obs, reward, cost, terminated, truncated, info = env.step(action)

            if not np.all(np.array(action) == 1):
                print(f'Action: {action}, reward: {reward:.2f}, cost: {cost:.2f}')

            rgb, mask, mixed = env.render()

            rgb_canvas.set_data(rgb)
            mask_canvas.set_data(mask)
            mixed_canvas.set_data(mixed)

            plt.draw()
            plt.pause(0.001)

            if terminated or truncated:
                env.reset()
    except KeyboardInterrupt:
        env.close()
