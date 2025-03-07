import os
import csv
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'stix'

font = {'family': 'STIXGeneral',
        'weight': 'bold',
        'size': 15,
        }

font_small = {'family': 'STIXGeneral',
              'weight': 'bold',
              'size': 12,
              }

font_large = {'family': 'STIXGeneral',
              'weight': 'bold',
              'size': 20,
              }


def read_done_reasons(filename: str):
    file_path = os.path.abspath(filename)
    print(file_path)
    assert os.path.exists(file_path)

    data = pd.read_csv(file_path)
    steps = data['TotalEnvSteps'].to_numpy()
    dr_collision = data['DoneReason/Collision'].to_numpy()
    dr_oovh = data['DoneReason/OutOfVolumeHorizontal'].to_numpy()
    dr_oovv = data['DoneReason/OutOfVolumeVertical'].to_numpy()
    dr_yaw = data['DoneReason/YawOverDeviation'].to_numpy()
    dr_idle = data['DoneReason/Idle'].to_numpy()
    dr_max_step = data['DoneReason/MaxStepsReached'].to_numpy()
    dr_success = data['DoneReason/Success'].to_numpy()
    # print(f'{steps=}')
    # print(f'{dr_oovh=}')
    return steps, dr_collision, dr_oovh, dr_oovv, dr_yaw, dr_idle, dr_max_step, dr_success


def read_ep_returns_costs(filename: str) -> (np.ndarray, np.ndarray):
    file_path = os.path.abspath(filename)
    print(file_path)
    assert os.path.exists(file_path)

    data = pd.read_csv(file_path)
    ep_ret = data['Metrics/EpRet'].to_numpy()
    ep_cost = data['Metrics/EpCost'].to_numpy()
    return ep_ret, ep_cost


def get_cost_rates(steps, dr_collision, dr_oovh, dr_oovv, dr_yaw, dr_idle, dr_max_step, dr_success):
    loose_cost_rate_arr = (dr_yaw + dr_idle + dr_max_step) / steps
    tight_cost_rate_arr = (dr_collision + dr_oovh + dr_oovv) / steps
    total_cost_rate_arr = loose_cost_rate_arr + tight_cost_rate_arr
    success_rate = dr_success / steps
    # return tight_cost_rate_arr, loose_cost_rate_arr, total_cost_rate_arr, success_rate
    return steps, tight_cost_rate_arr, loose_cost_rate_arr, total_cost_rate_arr


def save_algo_stat(file_paths: List[str]):
    assert len(file_paths) == 3

    epochs = None
    algo_ep_rets = []
    algo_ep_costs = []
    algo_tight_cost_rates = []
    algo_loose_cost_rates = []
    algo_total_cost_rates = []
    for filepath in file_paths:
        ep_ret, ep_cost = read_ep_returns_costs(filepath)
        steps, tight_cost_rate, loose_cost_rate, total_cost_rate = get_cost_rates(*read_done_reasons(filepath))
        if epochs is None:
            epochs = steps
        algo_ep_rets.append(ep_ret)
        algo_ep_costs.append(ep_cost)
        algo_tight_cost_rates.append(tight_cost_rate)
        algo_loose_cost_rates.append(loose_cost_rate)
        algo_total_cost_rates.append(total_cost_rate)

    algo_ep_rets = np.asarray(algo_ep_rets, dtype=np.float32)
    algo_ep_costs = np.asarray(algo_ep_costs, dtype=np.float32)
    algo_tight_cost_rates = np.asarray(algo_tight_cost_rates, dtype=np.float32)
    algo_loose_cost_rates = np.asarray(algo_loose_cost_rates, dtype=np.float32)
    algo_total_cost_rates = np.asarray(algo_total_cost_rates, dtype=np.float32)

    algo_ep_ret_mean = np.mean(algo_ep_rets, axis=0)
    algo_ep_ret_std = np.std(algo_ep_rets, axis=0)
    algo_ep_cost_mean = np.mean(algo_ep_costs, axis=0)
    algo_ep_cost_std = np.std(algo_ep_costs, axis=0)
    algo_tight_cost_rate_mean = np.mean(algo_tight_cost_rates, axis=0)
    algo_tight_cost_rate_std = np.std(algo_tight_cost_rates, axis=0)
    algo_loose_cost_rate_mean = np.mean(algo_loose_cost_rates, axis=0)
    algo_loose_cost_rate_std = np.std(algo_loose_cost_rates, axis=0)
    algo_total_cost_rate_mean = np.mean(algo_total_cost_rates, axis=0)
    algo_total_cost_rate_std = np.std(algo_total_cost_rates, axis=0)

    data = np.transpose(np.vstack((epochs, algo_ep_ret_mean, algo_ep_ret_std, algo_ep_cost_mean, algo_ep_cost_std,
                        algo_tight_cost_rate_mean, algo_tight_cost_rate_std,
                        algo_loose_cost_rate_mean, algo_loose_cost_rate_std,
                        algo_total_cost_rate_mean, algo_total_cost_rate_std)))
    df = pd.DataFrame(data, columns=['steps', 'ret_mean', 'ret_std', 'cost_mean', 'cost_std',
                                     'tight_cr_mean', 'tight_cr_std',
                                     'loose_cr_mean', 'loose_cr_std',
                                     'total_cr_mean', 'total_cr_std'])
    df.to_csv('stat.csv')
    print(f'Algo training stats saved!')


def plot_return_and_cost_rate(algos: List[str]):
    assert len(algos) > 0

    plt.style.use("seaborn-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for algo in algos:
        stat_filepath = './runs/' + algo + '-{Medium}/stat.csv'
        assert os.path.exists(stat_filepath), f'{stat_filepath} does not exist!'
        data = pd.read_csv(stat_filepath)
        steps = data['steps'].to_numpy()
        ret_mean = data['ret_mean'].to_numpy()
        ret_std = data['ret_std'].to_numpy()
        cr_mean = data['total_cr_mean'].to_numpy()
        cr_std = data['total_cr_std'].to_numpy()

        ax1.plot(steps, ret_mean, linewidth=2, label=algo)
        ax1.fill_between(steps, ret_mean - ret_std, ret_mean + ret_std, alpha=0.2)
        ax1.set_xlabel('Steps', fontdict=font)
        ax1.set_ylabel('Episodic Return', fontdict=font)
        ax1.set_xlim(0, 200000)
        leg = ax1.legend(loc="best", prop=font, ncol=2)
        for l in leg.get_lines():
            l.set_linewidth(4.0)

        ax2.plot(steps, cr_mean, linewidth=2, label=algo)
        ax2.fill_between(steps, cr_mean - cr_std, cr_mean + cr_std, alpha=0.2)
        ax2.set_xlabel('Steps', fontdict=font)
        ax2.set_ylabel('Cost Rate', fontdict=font)
        ax2.set_xlim(0, 200000)

    plt.tight_layout()
    plt.savefig('return_cost_rate_training.png', dpi=600)
    plt.show()


def plot_tight_loose_cost_rate(algos: List[str]):
    assert len(algos) > 0

    plt.style.use("seaborn-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for algo in algos:
        stat_filepath = './runs/' + algo + '-{Medium}/stat.csv'
        assert os.path.exists(stat_filepath), f'{stat_filepath} does not exist!'
        data = pd.read_csv(stat_filepath)
        steps = data['steps'].to_numpy()
        tight_cr_mean = data['tight_cr_mean'].to_numpy()
        tight_cr_std = data['tight_cr_std'].to_numpy()
        loose_cr_mean = data['loose_cr_mean'].to_numpy()
        loose_cr_std = data['loose_cr_std'].to_numpy()

        ax1.plot(steps, tight_cr_mean, linewidth=2, label=algo)
        ax1.fill_between(steps, tight_cr_mean - tight_cr_std, tight_cr_mean + tight_cr_std, alpha=0.2)
        ax1.set_xlabel('Steps', fontdict=font)
        ax1.set_ylabel('Tight Cost Rate', fontdict=font)
        ax1.set_xlim(0, 200000)
        leg = ax1.legend(loc=(0.4, 0.6), prop=font, ncol=2)
        for l in leg.get_lines():
            l.set_linewidth(4.0)

        ax2.plot(steps, loose_cr_mean, linewidth=2, label=algo)
        ax2.fill_between(steps, loose_cr_mean - loose_cr_std, loose_cr_mean + loose_cr_std, alpha=0.2)
        ax2.set_xlabel('Steps', fontdict=font)
        ax2.set_ylabel('Loose Cost Rate', fontdict=font)
        ax2.set_xlim(0, 200000)

    plt.tight_layout()
    plt.savefig('tight_loose_cost_rate_training.png', dpi=600)
    plt.show()



def plot_cost_rates(steps: np.ndarray, tight_cost_rates: np.ndarray, loose_cost_rates: np.ndarray, total_cost_rates: np.ndarray):
    if len(tight_cost_rates.shape) == 1:
        tight_cost_rates = tight_cost_rates[np.newaxis, ...]
    if len(loose_cost_rates.shape) == 1:
        loose_cost_rates = loose_cost_rates[np.newaxis, ...]
    if len(total_cost_rates.shape) == 1:
        total_cost_rates = total_cost_rates[np.newaxis, ...]

    tight_cost_rates_mean = np.mean(tight_cost_rates, axis=0)
    tight_cost_rates_std = np.std(tight_cost_rates, axis=0)

    loose_cost_rates_mean = np.mean(loose_cost_rates, axis=0)
    loose_cost_rates_std = np.std(loose_cost_rates, axis=0)

    total_cost_rates_mean = np.mean(total_cost_rates, axis=0)
    total_cost_rates_std = np.std(total_cost_rates, axis=0)

    plt.style.use("seaborn-darkgrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    ax1.plot(steps, tight_cost_rates_mean, color='tab:red',  label='algo')
    ax1.fill_between(steps, tight_cost_rates_mean - tight_cost_rates_std, tight_cost_rates_mean + tight_cost_rates_std,
                     color='red', alpha=0.2)
    # format x ticks
    xfmt = ticker.ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-6, -5))
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.get_xaxis().get_offset_text().set_visible(False)
    ax1_max = max(ax1.get_xticks())
    exponent_axis = np.floor(np.log10(ax1_max)).astype(int)
    ax1.annotate(r'$\times$10$^{%i}$' % exponent_axis, xy=(.91, .01), xycoords='axes fraction', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    ax1.set_title('Tight Cost Rate', fontdict=font)
    ax1.legend(loc="upper right", prop=font)

    ax2.plot(steps, loose_cost_rates_mean, color='tab:red',  label='algo')
    ax2.fill_between(steps, loose_cost_rates_mean - loose_cost_rates_std, loose_cost_rates_mean + loose_cost_rates_std,
                     color='red', alpha=0.2)
    ax2.set_title('Loose Cost Rate', fontdict=font)
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.get_xaxis().get_offset_text().set_visible(False)
    ax2_max = max(ax2.get_xticks())
    exponent_axis = np.floor(np.log10(ax2_max)).astype(int)
    ax2.annotate(r'$\times$10$^{%i}$' % exponent_axis, xy=(.91, .01), xycoords='axes fraction', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    ax3.plot(steps, total_cost_rates_mean, color='tab:red',  label='algo')
    ax3.fill_between(steps, total_cost_rates_mean - total_cost_rates_std, total_cost_rates_mean + total_cost_rates_std,
                     color='red', alpha=0.2)
    ax3.set_title('Total Cost Rate', fontdict=font)
    ax3.xaxis.set_major_formatter(xfmt)
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax3.get_xaxis().get_offset_text().set_visible(False)
    ax3_max = max(ax3.get_xticks())
    exponent_axis = np.floor(np.log10(ax3_max)).astype(int)
    ax3.annotate(r'$\times$10$^{%i}$' % exponent_axis, xy=(.91, .01), xycoords='axes fraction', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)

    ax1.set_xlim(0, 200000)
    ax2.set_xlim(0, 200000)
    ax3.set_xlim(0, 200000)
    plt.tight_layout()
    plt.show()


def envs_ret_cost_stats(algo: str, env_id: str):
    # algo_path = f'./eval/{algo}'
    # algo_path = f'./eval-V1/{algo}'
    algo_path = f'./eval-V2/{algo}'
    algo_env_path = algo_path + f'/{env_id}'
    assert os.path.exists(algo_env_path), f'{algo_env_path} does not exist!'

    rets = []
    costs = []
    lengths = []
    for seed in range(3):
        algo_env_seed_path = algo_env_path + f'/seed{seed}'
        assert os.path.exists(algo_env_seed_path), f'{algo_env_seed_path} does not exist!'

        result_path = algo_env_seed_path + '/result.txt'
        assert os.path.exists(result_path), f'{result_path} does not exist!'

        with open(result_path, 'r') as f:
            for line in f:
                if 'Episode' in line:
                    if 'reward' in line:
                        rets.append(float(line.split()[-1]))
                    if 'cost' in line:
                        costs.append(float(line.split()[-1]))
                    if 'length' in line:
                        lengths.append(float(line.split()[-1]))

    ret_mean = np.mean(rets)
    ret_std = np.std(rets)
    cost_mean = np.mean(costs)
    cost_std = np.std(costs)
    len_mean = np.mean(lengths)
    len_std = np.std(lengths)

    print(f'{algo=} {env_id=}')
    # print(f'{rets=}')
    # print(f'{costs=}')
    # print(f'{ret_mean=:.2f} {ret_std=:.2f} {cost_mean=:.2f} {cost_std=:.2f} {len_mean=:.1f} {len_std=:.1f}')
    print(f'{ret_mean=:.2f} {ret_std=:.2f} {cost_mean=:.2f} {cost_std=:.2f}')
    print('*' * 100)
    print('\r\n')


def plot_dr_bar(algos: List[str], eval_dir: str):
    assert os.path.exists(eval_dir), f'{eval_dir} does not exist!'

    algos_num = len(algos)
    assert algos_num > 0, f'No algorithm is given!'

    algo2dr_stat = {}
    for algo in algos:
        algo_eval_dir = os.path.join(eval_dir, algo)
        assert os.path.exists(algo_eval_dir), f'{algo_eval_dir} does not exist!'

        dr_dict: dict[int, int] = {0: 0,
                                   1: 0,
                                   2: 0,
                                   3: 0,
                                   4: 0,
                                   5: 0,
                                   6: 0}
        for level in ['Easy', 'Medium', 'Hard']:
        # for level in ['Medium']:
            for seed in range(3):
            # for seed in range(1):
                result_path = os.path.join(algo_eval_dir, level, f'seed{str(seed)}', 'result.txt')
                assert os.path.exists(result_path), f'{result_path} does not exist!'
                dr_found = False
                with open(result_path, 'r') as f:
                    for line in f:
                        if 'Done' in line:
                            dr_found = True
                            continue
                        if dr_found:
                            # print(f'{line=}')
                            # print(type(line[0]))
                            for dr in line:
                                if dr != ' ':
                                    dr_dict[int(dr)] += 1

        print(f'{dr_dict=}')
        algo2dr_stat[algo] = dr_dict

    # get default color sequence
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    xlabel_dict = {0: 'Collision',
                   1: 'OutOfVolumeHorizontally',
                   2: 'OutOfVolumeVertically',
                   3: 'YawOverDeviation',
                   4: 'Idle',
                   5: 'MaxStepReached'}

    plt.style.use("seaborn-darkgrid")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for done_reason, ax in zip(range(6), axes):
        algos_done_times = []
        for algo in algos:
            done_times = algo2dr_stat[algo][done_reason]
            algos_done_times.append(done_times)
        ax.bar(range(algos_num), algos_done_times, color=colors, label=algos)
        ax.get_xaxis().set_ticklabels([])
        ax.set_xlabel(xlabel_dict[done_reason], fontdict=font)
        if done_reason == 0:
            leg = ax.legend(loc="best", prop=font_small)
            for l in leg.get_lines():
                l.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig('done_reasons_bar.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    # fn = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/runs/DDPGLag-{Medium}/seed-000-2024-01-10-01-06-49/progress.csv'
    # plot_cost_rates(*get_cost_rates(*read_done_reasons(fn)))

    fn1 = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/runs/CCEPETS-{Medium}/seed-000-2024-01-13-16-03-56/progress.csv'
    fn2 = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/runs/CCEPETS-{Medium}/seed-001-2024-01-13-19-28-44/progress.csv'
    fn3 = '/home/edison/Research/ml-agents/ml-agents-envs/mlagents_envs/envs/runs/CCEPETS-{Medium}/seed-002-2024-01-13-21-58-03/progress.csv'
    # save_algo_stat([fn1, fn2, fn3])


    algos = ['PPO', 'PPOLag', 'FOCOPS', 'P3O', 'OnCRPO', 'DDPGLag', 'TD3Lag', 'SACLag', 'SafeLOOP', 'CCEPETS']
    # plot_return_and_cost_rate(algos)
    plot_tight_loose_cost_rate(algos)
    # plot_dr_bar(algos, './eval-V3')

    # for algo in algos:
    #     if algo == 'PPO':
    #         continue
    #
    #     algo_dir = './eval/' + algo
    #     assert os.path.exists(algo_dir)
    #     for level in ['Easy', 'Medium', 'Hard']:
    #         level_dir = algo_dir + '/' + level
    #         os.makedirs(level_dir)
    #         for seed in range(3):
    #             seed_dir = level_dir + '/' + 'seed' + str(seed)
    #             os.makedirs(seed_dir)

    # for algo in algos:
    #     for env in ['Easy', 'Medium', 'Hard']:
    #         envs_ret_cost_stats(algo, env)






