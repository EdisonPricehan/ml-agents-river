# Safe Riverine Environment 
<table>
  <tr>
    <td style="text-align: center;">
      <img src="images/unity-river-overview-min.png" width="500"><br>
      <strong>Sand Island</strong>
    </td>
    <td style="text-align: center;">
      <img src="images/unity-feature-bridge-min.png" width="500"><br>
      <strong>Bridge</strong>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="images/unity-feature-tributary-min.png" width="500"><br>
      <strong>Tributary</strong>
    </td>
    <td style="text-align: center;">
      <img src="images/unity-feature-varying-widths-min.png" width="500"><br>
      <strong>Varying widths & depths</strong>
    </td>
  </tr>
</table>



## Overview
The **Safe Riverine Environment (SRE)** is designed for vision-driven reinforcement learning in autonomous UAV river-following tasks.
It provides a photo-realistic riverine scene with structured navigation challenges, where the agent must follow a predefined river spline while avoiding obstacles such as bridges, while safety is enforced through explicit cost feedback with different severity levels to penalize unsafe behaviors like excessive deviation from the river path, prolonged idling, and collisions.
The environment is formulated as a **Partially Observable - Constrained Submodular Markov Decision Process (PO-CSMDP)** to balance task performance and safety for first-person-view coverage navigation.

<div style="text-align: center;">
  <img src="ml-agents-envs/images/river-spline-demo.png" alt="sre diagram" width="300" />
  <p><strong>SRE Diagram</strong></p>
</div>

## Observation Space
The agent receives a tuple of **RGB image** and **binary water semantic mask** of the drone view, of shape **[128, 128, 3]** and **[128, 128, 4]** respectively. 
You can further process the observation using either variational encoding or mask patchification to scale down to a lower dimension, depending on your task objective or preference.

<div style="text-align: center;">
  <img src="images/sre-observation-demo.gif" alt="sre obs demo" width="400" />
</div>


## Action Space
The agent operates in a **multi-discrete action space: MultiDiscrete([3,3,3,3])**, where each dimension represents a different movement axis, and each action is chosen from {0, 1, 2}:

- Axis 0 (Up-Down Translation): {0: Move Up, 1: No Operation, 2: Move Down}
- Axis 1 (Horizontal Rotation): {0: Rotate Left, 1: No Operation, 2: Rotate Right}
- Axis 2 (Longitudinal Translation - Forward/Backward): {0: Move Forward, 1: No Operation, 2: Move Backward}
- Axis 3 (Latitudinal Translation - Left/Right): {0: Move Left, 1: No Operation, 2: Move Right}

This waypoint-based control abstracts the UAV’s low-level dynamics while allowing flexible movement in all relevant spatial dimensions.

*Note*: The longitudinal backward translation is disabled in SRE to prevent agent learn dangerous backing strategy to gain rewards, meaning action like [1, 1, 2, 1] will not change agent position.

## Reward Function
The agent is rewarded based on its progress in covering the river spline:
- **1** for each newly visited river segment.
- **0** otherwise.

This **submodular reward structure (non-Markovian)** incentivizes exploration of unvisited areas while discouraging redundant actions.

## Cost Function
The cost function penalizes unsafe behaviors based on environmental hazards:
- **0.5** for minor violations (e.g., excessive yaw deviation, idling for too many steps).
- **1** for severe violations (e.g., leaving the river boundary, colliding with a bridge).

The cost function is **Markovian**, meaning it depends only on the current observation, ensuring timely safety feedback.

## Termination Conditions
An episode terminates if:
1. The agent **fully covers the river spline** (successful task completion).
2. The agent **exceeds safety constraints**, such as:
  - Leaving the defined river region.
  - Remaining idle for too long without exploring new river segments.
  - Colliding with obstacles such as bridges. 
  - The episode reaches the **maximum time limit**.

The agent will be randomly reset at a safe and valid pose above the river on episode begin.

The detailed statistics of count of done reasons will be displayed after the environment is closed.
Available done reasons are below:
```python
# Critical failure reasons
Collision = 0
OutOfVolumeHorizontal = 1
OutOfVolumeVertical = 2

# Loose failure reasons
YawOverDeviation = 3
Idle = 4
MaxStepReached = 5

# Success reason
Success = 6
```


## Additional Features
- **Difficulty Levels:** SRE includes multiple difficulty settings (easy, medium, hard), where higher difficulty introduces more complex river structures and obstacles.
- **SafeRL Integration:** The environment supports **Safe Reinforcement Learning (SafeRL)** algorithms by providing structured cost feedback for safety-aware policy training.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="images/river-easy-ortho.png" width="400"><br>
      <strong>Easy</strong>
    </td>
    <td style="text-align: center;">
      <img src="images/river-medium-ortho.png" width="400"><br>
      <strong>Medium</strong>
    </td>
    <td style="text-align: center;">
      <img src="images/river-hard-ortho.png" width="400"><br>
      <strong>Hard</strong>
    </td>
  </tr>
</table>

This structured environment serves as a benchmark for developing and evaluating **vision-driven autonomous navigation policies** with explicit safety constraints.

## Python Interface
The `safe-riverine-envs` Python package is built upon the
[ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 
Specifically, the `mlagents_envs` Python package. 

## Installation

Install the `safe-riverine-envs` package with:

```bash
python -m pip install safe-riverine-envs
```

Download the environments from this [link](https://purdue0-my.sharepoint.com/:f:/g/personal/wang5044_purdue_edu/EhGXA_oN1DpPoW-XlMKL2r8Bdv4Cu52tnQTzvNPQFwgawQ?e=QyHEgU) then unzip them.

## Usage
```python
from mlagents_envs.envs.env_utils import make_unity_env
import numpy as np


def run():
  """
  Apply random action to safe riverine environment
  """
  # Env path (change to you specific env location)
  env_path = '/home/edison/Research/unity-saferl-envs/medium_dr/riverine_medium_dr_env.x86_64'

  # Make env
  env = make_unity_env(env_path=env_path, max_idle_steps=50000)
  obs, info = env.reset()

  # Start the loop
  try:
    i = 0
    while i < 10000:
      action = env.action_space.sample()

      obs, reward, cost, terminated, truncated, info = env.step(action)

      if not np.all(np.array(action) == 1):
        print(f'Action: {action}, reward: {reward:.2f}, cost: {cost:.2f}')

      if terminated or truncated:
        env.reset()
  except KeyboardInterrupt:
    env.close()


if __name__ == '__main__':
  run()


```

If you find our work useful in your research, please cite our paper:
```
@article{wang2024vision,
  title={Vision-driven UAV River Following: Benchmarking with Safe Reinforcement Learning},
  author={Wang, Zihan and Mahmoudian, Nina},
  journal={IFAC-PapersOnLine},
  volume={58},
  number={20},
  pages={421--427},
  year={2024},
  publisher={Elsevier}
}
```

## Limitations

- `mlagents_envs` uses localhost ports to exchange data between Unity and
  Python. As such, multiple instances can have their ports collide, leading to
  errors. Make sure to use a different port if you are using multiple instances
  of `UnityEnvironment`.
- Communication between Unity and the Python `UnityEnvironment` is not secure.
- On Linux, ports are not released immediately after the communication closes.
  As such, you cannot reuse ports right after closing a `UnityEnvironment`.
