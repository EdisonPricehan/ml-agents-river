import os.path

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# plt.style.use("seaborn-darkgrid")
plt.style.use("seaborn-v0_8-darkgrid")


import seaborn as sns
import csv
import numpy as np
import ast
from typing import List
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import rotate
from PIL import Image


def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)


def calculate_endpoint(start, a, b, c, d):
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint


def calculate_endpoints(starts, quaternions):
    assert starts.shape[0] == quaternions.shape[0]
    endpoints = []
    for start, q in zip(starts, quaternions):
        endpoint = calculate_endpoint(start, *q)
        endpoints.append(endpoint)
    return np.array(endpoints)


def violin_plot(map_algo_list: List[List[str]]):
    num_difficulty = len(map_algo_list)
    assert num_difficulty > 0
    num_algorithm = len(map_algo_list[0])
    assert num_algorithm > 0

    all_map_algo_rew = []
    for algo_paths in map_algo_list:
        for algo_path in algo_paths:
            with open(algo_path, 'r') as f:
                reader = csv.reader(f)
                traj_len_rew = np.array(list(reader))
                # print(f'{traj_len_rew=}')
                traj_lens = np.array([ast.literal_eval(v) for v in traj_len_rew[:, 0]])
                traj_rews = np.array([ast.literal_eval(v) for v in traj_len_rew[:, 1]])

                # print(f'{traj_lens=}')
                # print(f'{traj_rews=}')
                # all_map_algo_rew.append(traj_rews / traj_lens)
                all_map_algo_rew.append(traj_rews)
                # all_map_algo_rew.append(traj_lens)
    # ep_rew_data = np.array(ep_rew_data)

    rews_medium_dynamic_jimmy = [4.4605265110731125, 2.7500001341104507, -0.21052628755569458, 10.065789833664894, 2.7500001341104507, 0.5131579488515854, 0.6447369009256363, 5.315789699554443, -0.6052631437778473, 5.447368651628494, -0.9342105239629745, 7.026316076517105, 1.3684211373329163, 0.3815789967775345, -1.0, 4.197368606925011, 10.065789833664894, 4.723684415221214, -0.9342105239629745, 5.381579175591469, 6.894737124443054, -0.21052628755569458, -0.5394736677408218, -1.0, 6.763158172369003, 2.2894738018512726, -0.7368420958518982, 2.6842106580734253, 3.8684212267398834, -0.5394736677408218, 5.578947603702545, -0.7368420958518982, 2.6184211820364, 3.0131580382585526, 2.7500001341104507, 0.5789474248886108, 2.4210527539253235, -0.5394736677408218, -0.07894733548164368, 2.8815790861845016, 0.05263161659240723, 2.815789610147476, -0.5394736677408218, -1.0, 8.605263501405716, 3.4078948944807053, -0.4736841917037964, -1.0, 3.7368422746658325, -0.9342105239629745]
    rews_medium_static_jimmy = [2.1578948497772217, 0.3815789967775345, 4.855263367295265, 6.105263411998749, 4.986842319369316, 4.06578965485096, 0.2500000447034836, 4.723684415221214, -0.14473681151866913, 3.6052633225917816, -0.9342105239629745, -0.7368420958518982, 1.3684211373329163, -0.9342105239629745, -1.0, -0.013157859444618225, 5.447368651628494, 2.6842106580734253, 1.3684211373329163, 0.18421056866645813, -0.40789471566677094, 2.5526317059993744, -1.0, -0.9342105239629745, 9.934210881590843, -0.07894733548164368, 0.18421056866645813, 0.31578952074050903, -0.9342105239629745, -0.7368420958518982, 6.5000002682209015, 2.355263277888298, 3.0131580382585526, -0.6710526198148727, 0.7763158529996872, 3.8684212267398834, 4.986842319369316, -0.8026315718889236, -0.6710526198148727, -0.8026315718889236, 2.6842106580734253, 10.000000357627869, 4.592105463147163, -1.0, 1.5657895654439926, 3.210526466369629, -0.27631576359272003, -1.0, 0.6447369009256363, -0.8684210479259491]

    rews_hard_dynamic_jimmy = [-0.5673076845705509, -0.5913461465388536, -0.9519230760633945, -1.0, 1.7884615883231163, -0.8076923042535782, -0.25480767898261547, 0.22596156038343906, -1.0, -0.5432692226022482, 0.5384615659713745, -0.8317307662218809, -1.0, -0.9759615380316973, 1.7884615864604712, 2.1250000558793545, -0.47115383855998516, -0.5673076845705509, 2.918269282206893, -0.8798076901584864, 2.076923131942749, -0.03846152313053608, -0.6394230704754591, -0.6394230704754591, -0.9759615380316973, -0.9759615380316973, 0.39423079416155815, 1.0432692673057318, 1.8605769742280245, 0.7307692598551512, -0.5192307606339455, -0.3749999888241291, -0.3509615324437618, -0.39903845079243183, -0.9038461521267891, -0.1826922930777073, -0.8076923042535782, -0.9759615380316973, -1.0, -0.6874999944120646, -0.11057690717279911, -0.8798076901584864, -1.0, -0.7836538422852755, -0.32692306488752365, -0.11057690717279911, -0.9519230760633945, 0.85096157155931, -0.6634615324437618, -0.8798076901584864]
    rews_hard_static_jimmy = [-0.9038461521267891, -0.9038461521267891, -0.6634615324437618, -1.0, 1.884615434333682, -0.8076923042535782, -0.39903845079243183, -0.1586538329720497, -1.0, -0.03846152313053608, -0.7596153803169727, -0.6874999944120646, -1.0, -0.9519230760633945, -0.4471153747290373, 1.283653886988759, -0.8076923042535782, 1.5721154287457466, -0.6153846085071564, -0.9519230760633945, 0.8990384954959154, -0.3509615268558264, -0.4951923005282879, -0.8076923042535782, -0.6874999944120646, -0.9519230760633945, 1.019230805337429, -0.6874999944120646, 1.9807692840695381, 0.2500000223517418, -1.0, -0.5432692226022482, -0.47115384228527546, -0.9278846140950918, -0.3509615268558264, -0.3509615287184715, -0.73557691834867, -0.9038461521267891, -1.0, -0.49519229866564274, -0.20673075504601002, -0.9759615380316973, -0.9759615380316973, -0.3509615268558264, -0.6874999944120646, -0.1826922930777073, 2.052884668111801, 0.3701923321932554, -0.6394230704754591, -0.6874999944120646]

    # all_map_algo_rew[2] = rews_medium_static_jimmy
    # all_map_algo_rew[3] = rews_medium_dynamic_jimmy
    # all_map_algo_rew[6] = rews_hard_static_jimmy
    # all_map_algo_rew[7] = rews_hard_dynamic_jimmy

    # CliffCircular env
    # PPO
    all_map_algo_rew[0] = [4.0, 17.0, 10.0, 2.0, 5.0, 2.0, 2.0, 8.0, 2.0, 7.0, 5.0, 2.0, 4.0, 3.0, 3.0, 2.0, 5.0, 3.0, 6.0, 11.0, 6.0, 3.0, 10.0, 5.0, 4.0, 13.0, 5.0, 2.0, 3.0, 5.0, 5.0, 2.0, 4.0, 9.0, 3.0, 2.0, 5.0, 6.0, 2.0, 8.0, 5.0, 8.0, 2.0, 2.0, 10.0, 8.0, 5.0, 2.0, 3.0, 3.0, 6.0, 2.0, 3.0, 2.0, 2.0, 2.0, 5.0, 2.0, 2.0, 11.0, 10.0, 2.0, 4.0, 2.0, 4.0, 5.0, 13.0, 2.0, 2.0, 4.0, 5.0, 10.0, 2.0, 8.0, 6.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 5.0, 3.0, 4.0, 5.0, 4.0, 2.0, 7.0, 2.0, 13.0, 4.0, 8.0, 3.0, 5.0, 2.0, 4.0, 4.0, 12.0, 2.0, 17.0]
    # BC
    all_map_algo_rew[1] = [20.0, -94.0, -88.0, -100.0, 18.0, 20.0, 20.0, 13.0, 20.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 14.0, 20.0, 20.0, 20.0, 20.0, 13.0, 20.0, 20.0, 13.0, 20.0, 20.0, -83.0, -91.0, 20.0, 20.0, 20.0, 20.0, 13.0, 18.0, 20.0, 19.0, 20.0, -95.0, 19.0, -99.0, 20.0, 20.0, 20.0, -97.0, 20.0, -83.0, -92.0, -100.0, 20.0, 20.0, 20.0, 20.0, -99.0, 20.0, -92.0, 20.0, 20.0, 20.0, 20.0, 20.0, 7.0, 19.0, 20.0, 20.0, 20.0, -91.0, 20.0, 20.0, 20.0, 20.0, 20.0, 8.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, -99.0, 20.0, -100.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, -92.0, -88.0, 20.0, 18.0, 20.0, 20.0, 20.0, 19.0, 13.0, 8.0]
    # PPO+StaticBC
    all_map_algo_rew[2] = [18.0, 19.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 16.0, 9.0, 15.0, 20.0, 20.0, 14.0, 20.0, 20.0, 13.0, 20.0, -83.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, -81.0, 20.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, -99.0, 20.0, 20.0, 20.0, -83.0, 18.0, 18.0, 20.0, 16.0, 20.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 16.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, -88.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 16.0, 10.0, 15.0, 20.0, 20.0, 20.0]
    # PPO+DynamicBC
    all_map_algo_rew[3] = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 5.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 18.0, 18.0, 13.0, 20.0, 20.0, 13.0, 20.0, 20.0, 20.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 16.0, 20.0, 20.0, 20.0, 13.0, 20.0, 20.0, 20.0, 20.0, 3.0, 20.0, 20.0, 18.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 8.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 11.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]

    # riverine env
    # all_map_algo_rew[5] = [-1.0, -0.6052631437778473, -0.9342105239629745, 5.447368651628494, 0.11842109262943268, -0.9342105239629745, 0.31578952074050903, -0.27631576359272003, 0.3815789967775345, -0.7368420958518982, 2.0921053737401962, 3.210526466369629, -0.9342105239629745, 0.2500000447034836, -0.7368420958518982, 2.1578948497772217, -0.6052631437778473, -0.013157859444618225, 1.631579041481018, -0.6710526198148727, 0.9078948050737381, 1.631579041481018, 0.5789474248886108, 1.039473757147789, -1.0, 0.31578952074050903, -0.9342105239629745, -1.0, 3.539473846554756, -1.0, -0.3421052396297455, 1.1052632331848145, 0.3815789967775345, 0.11842109262943268, 2.223684325814247, -0.9342105239629745, -1.0, -0.8026315718889236, -0.21052628755569458, 4.78947389125824, -1.0, 3.7368422746658325, -0.5394736677408218, 0.3815789967775345, 2.0921053737401962, -0.8026315718889236, 2.4210527539253235, 2.6842106580734253, -0.27631576359272003, 5.118421271443367]
    all_map_algo_rew[6] = rews_medium_static_jimmy
    all_map_algo_rew[7] = rews_medium_dynamic_jimmy

    medium_algo_rew = all_map_algo_rew[:num_algorithm]
    # print(f'{len(medium_algo_rew)=}')
    if len(all_map_algo_rew) > num_algorithm:
        hard_algo_rew = all_map_algo_rew[num_algorithm:]

    medium_mins = [min(algo_rews) for algo_rews in medium_algo_rew]
    medium_maxs = [max(algo_rews) for algo_rews in medium_algo_rew]
    medium_quartile1, medium_medians, medium_quartile3 = np.percentile(medium_algo_rew, [25, 50, 75], axis=1)

    hard_mins = [min(algo_rews) for algo_rews in hard_algo_rew]
    hard_maxs = [max(algo_rews) for algo_rews in hard_algo_rew]
    hard_quartile1, hard_medians, hard_quartile3 = np.percentile(hard_algo_rew, [25, 50, 75], axis=1)

    print(f'{len(all_map_algo_rew)=}')

    # plt.figure(figsize=(30, 18))
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 30,
            }
    fs = 35
    algo_labels = ['PPO', 'BC', 'PPO+BC-S', 'PPO+BC-D']
    colors = ['orange', 'green', 'blue', 'red']
    alpha = 0.3

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(26, 12))
    plt.rcParams['font.weight'] = 'bold'
    # ax1.set_xticklabels(algo_labels, fontsize=fs, fontweight='bold', fontdict=font)
    ax1.set_xticklabels(algo_labels, fontdict=font)
    ax1.set_xticks(np.arange(1, num_algorithm + 1), labels=algo_labels)
    ax1.set_xlabel('Algorithms', fontdict=font)
    # ax1.set_yticks(np.arange(-1, 11, 2))
    ax1.set_yticks(np.arange(-100, 21, 20))
    plt.yticks(fontsize=fs)
    # ax1.set_title('Training Map', font)
    ax1.set_title('CliffCircular-gym-v0 Env', font)
    ax1.set_ylabel('Episode Reward', fontdict=font)
    ax1.tick_params(axis='y', which='major', labelsize=fs)

    parts = ax1.violinplot(medium_algo_rew, showmeans=True, showextrema=False, showmedians=False, bw_method=0.5)
    print(dir(parts['cmeans']))
    parts['cmeans'].set_linewidths(2)
    parts['cmeans'].set_colors('k')
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(alpha)

    ax1.scatter(np.arange(1, num_algorithm + 1), medium_medians, marker='o', color='white', s=40, zorder=3)
    ax1.vlines(np.arange(1, num_algorithm + 1), medium_quartile1, medium_quartile3, color='k', linestyle='-', lw=10)
    ax1.vlines(np.arange(1, num_algorithm + 1), medium_mins, medium_maxs, color='k', linestyle='-', lw=1)

    plt.rcParams['font.weight'] = 'bold'
    ax2.set_xticklabels(algo_labels, fontdict=font)
    ax2.set_xticks(np.arange(1, num_algorithm + 1), labels=algo_labels)
    ax2.yaxis.tick_right()
    # ax2.set_yticks(np.arange(-1, 6))
    ax2.set_yticks(np.arange(-1, 11))
    plt.yticks(fontsize=fs)
    ax2.set_xlabel('Algorithms', fontdict=font)
    # ax2.set_title('Testing Map', font)
    ax2.set_title('Unity-riverine Env', font)
    parts = ax2.violinplot(hard_algo_rew, showmeans=True, showextrema=False, showmedians=False, bw_method=0.5)
    print(dir(parts['cmeans']))
    parts['cmeans'].set_linewidths(2)
    parts['cmeans'].set_colors('k')
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(alpha)
    ax2.scatter(np.arange(1, num_algorithm + 1), hard_medians, marker='o', color='white', s=40, zorder=3)
    ax2.vlines(np.arange(1, num_algorithm + 1), hard_quartile1, hard_quartile3, color='k', linestyle='-', lw=10)
    ax2.vlines(np.arange(1, num_algorithm + 1), hard_mins, hard_maxs, color='k', linestyle='-', lw=1)

    custom_lines = [Line2D([0], [0], color=colors[0], lw=20, alpha=alpha),
                    Line2D([0], [0], color=colors[1], lw=20, alpha=alpha),
                    Line2D([0], [0], color=colors[2], lw=20, alpha=alpha),
                    Line2D([0], [0], color=colors[3], lw=20, alpha=alpha)]
    ax2.legend(custom_lines, ['PPO', 'BC', 'PPO+StaticBC', 'PPO+DynamicBC'], loc='upper left', prop=font)

    # for i in range(num_algorithm):
    #     parts = plt.violinplot(all_map_algo_rew[i::num_algorithm], list(range(num_difficulty)), showmeans=True, showextrema=False, showmedians=False, bw_method=0.5)
    #
    #     for pc in parts['bodies']:
    #         # pc.set_facecolor('#D43F3A')
    #         pc.set_edgecolor('black')
    #         pc.set_alpha(0.2)

    # plt.legend(['PPO', 'BC', 'PPO+StaticBC', 'PPO+DynamicBC'], fontsize=30, loc='upper left')
    plt.tight_layout()
    plt.savefig('violin', dpi=100)
    # plt.show()


def bar_plot(map_algo_list: List[List[str]]):
    num_difficulty = len(map_algo_list)
    assert num_difficulty > 0
    num_algorithm = len(map_algo_list[0])
    assert num_algorithm > 0

    all_map_algo_rew = []
    for algo_paths in map_algo_list:
        for algo_path in algo_paths:
            with open(algo_path, 'r') as f:
                reader = csv.reader(f)
                traj_len_rew = np.array(list(reader))
                traj_lens = np.array([ast.literal_eval(v) for v in traj_len_rew[:, 0]])
                traj_rews = np.array([ast.literal_eval(v) for v in traj_len_rew[:, 1]])

                # print(f'{traj_lens=}')
                # print(f'{traj_rews=}')
                all_map_algo_rew.append(traj_rews / traj_lens)
    # ep_rew_data = np.array(ep_rew_data)

    print(f'{len(all_map_algo_rew)=}')
    all_step_rew_mean = [np.mean(item) for item in all_map_algo_rew]
    all_step_rew_std = [np.std(item) for item in all_map_algo_rew]

    width = 0.25

    plt.figure(figsize=(30, 18))
    # for i in range(num_algorithm):
    plt.bar(np.arange(num_difficulty), all_step_rew_mean[0::2], yerr=all_step_rew_std[0::2], align='center', alpha=0.5, ecolor='black', capsize=10, width=width, label='PPO+StaticBC')
    plt.bar(np.arange(num_difficulty) + width, all_step_rew_mean[1::2], yerr=all_step_rew_std[1::2], align='center', alpha=0.5, ecolor='black', capsize=10,  width=width, label='PPO+DynamicBC')

    plt.legend()
    plt.show()


def spline_plot(river_splines_csv: List[str], show: bool = False, plot_central: bool = False):
    for spline_path in river_splines_csv:
        assert os.path.exists(spline_path)

    central_path, up_path, down_path = river_splines_csv
    central_points, up_points, down_points = [], [], []

    with open(central_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for point in reader:
            x = ast.literal_eval(point[0])
            y = ast.literal_eval(point[1])
            central_points.append([x, y])
    central_points = np.array(central_points)

    with open(up_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for point in reader:
            x = ast.literal_eval(point[0])
            y = ast.literal_eval(point[1])
            up_points.append([x, y])
    up_points = np.array(up_points)

    with open(down_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for point in reader:
            x = ast.literal_eval(point[0])
            y = ast.literal_eval(point[1])
            down_points.append([x, y])
    down_points = np.array(down_points)

    plt.figure(figsize=(30, 18))

    # img_path = '/home/edison/Research/Paper Images/Drone Along River/reworked/river medium ortho.png'
    # images = Image.open(img_path)
    # img_arr = np.array(images)
    # rotated_img_arr = rotate(img_arr, -90)
    # plt.imshow(rotated_img_arr, extent=[0, 120, 0, 120])

    if plot_central:
        plt.plot(central_points[:, 0], central_points[:, 1], c='b')
        plt.scatter(central_points[:, 0], central_points[:, 1], c='r')

    # plt.plot(up_points[:, 0], up_points[:, 1], c='cyan', linestyle='dashed', linewidth=4)
    plt.plot(up_points[:, 0], up_points[:, 1], c='cyan', linewidth=8)
    # plt.plot(down_points[:, 0], down_points[:, 1], c='cyan', llinestyle='dashed', linewidth=4)
    plt.plot(down_points[:, 0], down_points[:, 1], c='cyan', linewidth=8)

    if show:
        plt.show()


def get_traj_dist(traj: np.ndarray) -> float:
    num = traj.shape[0]
    if num <= 1:
        return 0

    total_dist = 0
    for i in range(num - 1):
        p1 = traj[i]
        p2 = traj[i + 1]
        dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        total_dist += dist
    return total_dist


def trajs_plot(traj_path_list: List[str]):
    small_dist = 5
    medium_dist = 80
    large_dist = 150

    colors = ['black', 'magenta', 'cyan', 'green']

    for traj_path in traj_path_list:
        points2d = []
        with open(traj_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                x = ast.literal_eval(line[0])
                y = ast.literal_eval(line[1])
                points2d.append([x, y])
        points2d = np.array(points2d)

        total_dist = get_traj_dist(points2d)
        # print(f'{total_dist=}')
        plt.plot(points2d[0, 0], points2d[0, 1], color='g', marker='o', markersize=20)

        if total_dist < small_dist:
            color = colors[0]
        elif total_dist < medium_dist:
            color = colors[1]
        elif total_dist < large_dist:
            color = colors[2]
        else:
            color = colors[3]
        plt.plot(points2d[:, 0], points2d[:, 1], color=color, linewidth=3)

        plt.plot(points2d[-1, 0], points2d[-1, 1], color='r', marker='X', markersize=20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('x (m)', fontsize=20)
    plt.ylabel('y (m)', fontsize=20)
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=20),
                    Line2D([0], [0], marker='X', color='w', markerfacecolor='r', markersize=20),
                    Line2D([0], [0], color='b', linestyle='--', lw=4),
                    Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[1], lw=4),
                    Line2D([0], [0], color=colors[2], lw=4),
                    Line2D([0], [0], color=colors[3], lw=4)]
    plt.legend(custom_lines,
               [f'Start',
                f'End',
                f'River bank',
                f'Length < {small_dist}',
                f'{small_dist} < Length < {medium_dist}',
                f'{medium_dist} < Length < {large_dist}',
                f'Length > {large_dist}'],
               loc='center', fontsize=30)
    # plt.title('Trajectories with random spawn')

    plt.show()


def get_traj_from_file(traj_file: str) -> np.ndarray:
    assert os.path.exists(traj_file)

    points2d = []
    with open(traj_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            x = ast.literal_eval(line[0])
            y = ast.literal_eval(line[1])
            points2d.append([x, y])
    points2d = np.array(points2d)
    return points2d


def trajs_plot_transparency(traj_dynamic_paths: List[str], traj_static_paths: List[str], plot_legend: bool = False):
    trajs_dynamic = []
    trajs_static = []
    assert len(traj_dynamic_paths) == len(traj_static_paths)

    for i in range(len(traj_dynamic_paths)):
        trajs_dynamic.append(get_traj_from_file(traj_dynamic_paths[i]))
        trajs_static.append(get_traj_from_file(traj_static_paths[i]))

    dists_dynamic = [get_traj_dist(traj) for traj in trajs_dynamic]
    dists_static = [get_traj_dist(traj) for traj in trajs_static]
    # print(f'{dists_dynamic=}')
    # print(f'{dists_static=}')
    max_dist = np.max([dists_dynamic, dists_static])
    # print(f'{max_dist=}')

    fs = 50
    lw = 5
    ms = 30
    check_point_steps = 50
    font = {'family': 'STIXGeneral',
            'weight': 'bold',
            'size': 45,
            }

    for traj_d, traj_s in zip(trajs_dynamic, trajs_static):
        dist_to_start_dyn = [get_traj_dist(traj_d[:i]) for i in range(1, len(traj_d) + 1)]
        alphas_dyn = dist_to_start_dyn / max_dist
        alphas_dyn = np.clip(alphas_dyn, a_min=0.2, a_max=1)
        dist_to_start_sta = [get_traj_dist(traj_s[:i]) for i in range(1, len(traj_s) + 1)]
        alphas_sta = dist_to_start_sta / max_dist
        alphas_sta = np.clip(alphas_sta, a_min=0.2, a_max=1)

        assert len(dist_to_start_dyn) == len(traj_d)
        assert len(dist_to_start_sta) == len(traj_s)

        # print(f'{alphas_dyn=}')
        # print(f'{alphas_sta=}')

        for s in range(1, len(alphas_dyn)):
            plt.plot(traj_d[0, 0], traj_d[0, 1], color='r', marker='o', markersize=ms, markerfacecolor='r', markerfacecoloralt='b', fillstyle='left')
            plt.plot(traj_d[s - 1: s + 1, 0], traj_d[s - 1: s + 1, 1], alpha=alphas_dyn[s], color='r', linewidth=lw)
            plt.plot(traj_d[-1, 0], traj_d[-1, 1], color='r', marker='X', markersize=ms)
            if s % check_point_steps == 0:
                plt.plot(traj_d[s, 0], traj_d[s, 1], color='r', marker='*', markersize=ms)

        for s in range(1, len(alphas_sta)):
            # plt.plot(traj_s[0, 0], traj_s[0, 1], color='b', marker='o', markersize=ms)
            plt.plot(traj_s[s - 1: s + 1, 0], traj_s[s - 1: s + 1, 1], alpha=alphas_sta[s], color='b', linewidth=lw)
            plt.plot(traj_s[-1, 0], traj_s[-1, 1], color='b', marker='X', markersize=ms)
            if s % check_point_steps == 0:
                plt.plot(traj_s[s, 0], traj_s[s, 1], color='b', marker='*', markersize=ms)

    custom_lines = [Line2D([0], [0], color='cyan', lw=2*lw),
                    Line2D([0], [0], color='r', lw=2*lw),
                    Line2D([0], [0], color='b', lw=2*lw),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markerfacecoloralt='b', markersize=1.5*ms, fillstyle='left'),
                    Line2D([0], [0], marker='X', color='w', markerfacecolor='r', markerfacecoloralt='b', markersize=1.5*ms, fillstyle='left'),
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markerfacecoloralt='b', markersize=2*ms, fillstyle='left')
                    ]

    if plot_legend:
        plt.legend(custom_lines,
                   [f'River Bank',
                    f'PPO+DynamicBC',
                    f'PPO+StaticBC',
                    f'Start',
                    f'End',
                    f'Checkpoint'
                    ],
                   loc='center', prop=font)

    plt.xlabel('x (m)', fontdict=font)
    plt.ylabel('y (m)', fontdict=font)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.title('Trajectories Comparison', fontdict=font)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    # with open(traj_path, 'r') as f:
    #     reader = csv.reader(f, delimiter=',')
    #     traj_random = list(reader)
    #
    # # print(f'{traj_random=}')
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # xs = [float(pose[0]) for pose in traj_random]
    # ys = [float(pose[1]) for pose in traj_random]
    # zs = [float(pose[2]) for pose in traj_random]
    # ax.plot3D(xs, zs, ys, 'red', linewidth=2)
    # # ax.scatter3D(xs, zs, ys, c=zs, cmap='Greens')
    #
    # start_points = np.array([xs, zs, ys]).transpose()
    # quaternions = np.array(traj_random).astype(float)[:, 3:]
    # print(f'{start_points.shape=} {quaternions.shape=}')
    # end_points = calculate_endpoints(start_points, quaternions)
    #
    # ax.quiver(xs, zs, ys,
    #           end_points[:, 0], end_points[:, 1], end_points[:, 2], length=1, normalize=True)
    #
    # plt.show()

    # difficulty_level = 'easy'
    difficulty_level = 'medium'
    # difficulty_level = 'hard'

    # algorithm = 'ppo_static_bc'
    algorithm = 'ppo_dynamic_bc'

    paths_all = []
    # for level in ['easy', 'medium', 'hard']:
    # for level in ['medium']:
    for level in ['medium', 'medium']:
    # for level in ['medium', 'hard']:
        paths = []
        for algo in ['ppo', 'bc', 'ppo_static_bc', 'ppo_dynamic_bc']:
        # for algo in ['bc', 'ppo_dynamic_bc']:
            path = f'river_splines/{level}/{algo}/stats.csv'
            paths.append(path)
        paths_all.append(paths)

    violin_plot(paths_all)
    # bar_plot(paths_all)

    spline_central_path = f'river_splines/{difficulty_level}/spline_central.csv'
    spline_up_path = f'river_splines/{difficulty_level}/spline_up.csv'
    spline_down_path = f'river_splines/{difficulty_level}/spline_down.csv'
    spline_plot([spline_central_path, spline_up_path, spline_down_path], show=False, plot_central=False)

    # traj_paths = []
    # n = 30
    # for root_dir, dirs, files in os.walk(f'river_splines/{difficulty_level}/traj_random'):
    #     i = 0
    #     for f in files:
    #         traj_paths.append(os.path.join(root_dir, f))
    #         i += 1
    #         if i >= n:
    #             break
    #
    # print(f'Total {len(traj_paths)} trajectories.')
    #
    # trajs_plot(traj_paths)

    '''
    Plot traj with transparency
    '''
    static_paths = []
    dynamic_paths = []
    # target_traj_ids = [55, 65, 74]
    # target_traj_ids = [55, 65]
    # target_traj_ids = [65, 74]
    # target_traj_ids = [65]
    target_traj_ids = [74]
    level = 'medium'
    for traj_id in target_traj_ids:
        dyn_path = f'river_splines/{level}/ppo_dynamic_bc/traj_random/traj{traj_id - 1}.csv'
        dynamic_paths.append(dyn_path)
        sta_path = f'river_splines/{level}/ppo_static_bc/traj_random/traj{traj_id - 1}.csv'
        static_paths.append(sta_path)

    # trajs_plot_transparency(dynamic_paths, static_paths, plot_legend=True)



