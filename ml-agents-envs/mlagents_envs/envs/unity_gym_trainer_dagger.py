import tempfile

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util import logger as imit_logger

from env_utils import make_unity_env

import sys

sys.path.append("/home/edison/Research/Mutual_Imitaion_Reinforcement_Learning")
from ppo import PPO

vae_model_name = 'vae-sim-rgb-easy.pth'


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.reward_threshold = 300
        self.reward_max = 300

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print("callback", self.n_calls)

            # reward, _, traj_good, traj_bad = evaluate_policy(
            #     self.model,  # type: ignore[arg-type]
            #     self.training_env,
            #     n_eval_episodes=3,
            #     render=False,
            #     reward_threshold=min(self.reward_max + 50, 500)
            # )
            # print("Mean PPO reward", reward)

            if self.n_calls % self.check_freq == 0:
                print(f'Saving bc weights, {self.n_calls=}!')
                # self.model.save("weight/" + rl_name + "_" + env_name + '_seed_'+str(seeds[i])+'_' + str(self.n_calls) + '_{:.4f}'.format(reward))
                self.model.save('weights/' + 'bc_' + str(self.n_calls) + '_{:3d}'.format(int(self.n_calls // self.check_freq)))

        return True


def train(seed: int = 0):
    expert_policy_path = 'policies/PPO_unity_river_100000_seed_3.zip'
    model_save_name = 'circular_easy_ppo_dagger'
    tb_log_dir = './ppo_river_tensorboard/'
    tb_log_name = 'easy_ppo_dagger_1'
    train_steps = 100000

    # env_path = '/home/edison/Terrain/terrain_rgb.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action1d.x86_64'
    # env_path = '/home/edison/Terrain/terrain_rgb_action4d.x86_64'
    env_path = '/home/edison/Terrain/circular_river_easy/circular_river_easy.x86_64'
    env = make_unity_env(env_path, 1, True, seed, vae_model_name=vae_model_name)

    rng = np.random.default_rng(seed)

    expert = PPO.load(expert_policy_path)

    logger = imit_logger.configure(tb_log_dir + tb_log_name, ["stdout", "csv", "tensorboard"])

    callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir='', verbose=1)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=64,
        custom_logger=logger,
        rng=rng,
    )

    cur_best_bc_mean_reward = -2

    def save_model_callback():
        reward, _ = evaluate_policy(bc_trainer.policy, env, 5)
        nonlocal cur_best_bc_mean_reward
        if reward > cur_best_bc_mean_reward:
            cur_best_bc_mean_reward = reward
            bc_trainer.save_policy(f'weights/dagger_reward_{cur_best_bc_mean_reward:.02f}')
            print(f'Saved new policy!')

    with tempfile.TemporaryDirectory(prefix="dagger_river_easy_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            custom_logger=logger,
            rng=rng,
        )
        dagger_trainer.allow_variable_horizon = True
        dagger_trainer.train(train_steps, rollout_round_min_episodes=3, rollout_round_min_timesteps=1024,
                             bc_train_kwargs={'n_epochs': 1,
                                              'log_interval': 10,
                                              'log_rollouts_n_episodes': 10,
                                              'on_epoch_end': save_model_callback(),
                                              })
        dagger_trainer.save_policy(model_save_name)
        print(f'Dagger policy is saved as {model_save_name}')

    reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
    print("Reward:", reward)


if __name__ == '__main__':
    train(seed=3)

