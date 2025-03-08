from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

# from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig
from mlagents_envs.side_channel.agent_configuration_channel import AgentConfigurationChannel, AgentConfig
from mlagents_envs.side_channel.agent_reset_channel import AgentResetChannel

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def make_unity_env(
    env_path: str,
    worker_id: int = 0,
    seed: int = 0,
    max_idle_steps: int = 50,
    # TODO add more configurable arguments
):
    # width, height = 512, 512  # tunable
    width, height = 128, 128  # tunable for manual control
    channel_env = EnvironmentParametersChannel()
    channel_env.set_float_parameter("simulation_mode", 1.0)
    channel_eng = EngineConfigurationChannel()
    channel_eng.set_configuration_parameters(width=width, height=height, quality_level=1, time_scale=1,
                                             target_frame_rate=50, capture_frame_rate=50)
    channel_agent = AgentConfigurationChannel()
    channel_agent.set_configuration_parameters(max_idle_steps=max_idle_steps)
    channel_reset = AgentResetChannel()

    print(f'Starting unity env ...')
    unity_env = UnityEnvironment(file_name=env_path, worker_id=worker_id, no_graphics=False, seed=seed,
                                 side_channels=[channel_env, channel_eng, channel_agent, channel_reset],
                                 additional_args=['-logFile', 'unity.log'])

    env = UnityToGymWrapper(unity_env,
                            uint8_visual=True,
                            flatten_branched=False,
                            allow_multiple_obs=True,
                            safe_rl=True,
                            )

    return env


# def make_unity_vec_env(env_directory: str,
#                        num_env: int,
#                        visual: bool,
#                        seed: int,
#                        start_index: int = 0,
#                        flatten_discrete_action: bool = False,
#                        safer_l: bool = False,
#                        no_graphics: bool = False,
# ):
#     """
#     Create a wrapped, monitored vector Unity environment.
#     """
#
#     def make_env(rank, seed, use_visual=True):  # pylint: disable=C0111
#         def _thunk():
#             width, height = 128, 128
#             channel_env = EnvironmentParametersChannel()
#             channel_env.set_float_parameter("simulation_mode", 1.0)
#
#             channel_eng = EngineConfigurationChannel()
#             channel_eng.set_configuration_parameters(width=width, height=height, quality_level=1, time_scale=1,
#                                                      target_frame_rate=50, capture_frame_rate=50)
#
#             channel_agent = AgentConfigurationChannel()
#             channel_agent.set_configuration_parameters(max_idle_steps=0)  # set to 0 to disable idle reset
#
#             channel_reset = AgentResetChannel()
#             channel_reset.set_reset_pose(random_reset=True)
#
#             unity_env = UnityEnvironment(env_directory, base_port=5000 + rank, seed=seed, no_graphics=no_graphics,
#                                          side_channels=[channel_env, channel_eng, channel_agent, channel_reset])
#
#             env = UnityToGymWrapper(unity_env, uint8_visual=use_visual, flatten_branched=flatten_discrete_action,
#                                     allow_multiple_obs=True, action_space_seed=seed, safe_rl=safer_l)
#
#             # new_logger = configure("/tmp/unity_sb3_ppo_log/", ["stdout", "csv", "tensorboard"])
#             # env = Monitor(env, filename=new_logger.get_dir())
#             # env = Monitor(env)
#             # env = RolloutInfoWrapper(env)
#             return env
#
#         return _thunk
#
#     if visual:
#         return SubprocVecEnv([make_env(i + start_index, seed) for i in range(num_env)])
#     else:
#         rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
#         return DummyVecEnv([make_env(rank, use_visual=False, seed=seed)])
