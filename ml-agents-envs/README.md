# Safe Riverine Environment Python Interface

The `safe-riverine-envs` Python package is built upon the
[ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). 
Specifically, the `mlagents_envs` Python package. 

## Installation

Install the `safe-riverine-envs` package with:

```bash
python -m pip install safe-riverine-envs
```

## Usage & More Information


## Limitations

- `mlagents_envs` uses localhost ports to exchange data between Unity and
  Python. As such, multiple instances can have their ports collide, leading to
  errors. Make sure to use a different port if you are using multiple instances
  of `UnityEnvironment`.
- Communication between Unity and the Python `UnityEnvironment` is not secure.
- On Linux, ports are not released immediately after the communication closes.
  As such, you cannot reuse ports right after closing a `UnityEnvironment`.
