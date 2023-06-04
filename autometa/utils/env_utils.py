from typing import Callable, Union

import torch
import gym

from gym.envs.registration import register
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from autometa.envs.multiprocessing_vec_env import MultiprocessingVecEnv
from autometa.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper
from autometa.envs.rl_squared_env import RLSquaredEnv


def get_vec_normalize(venv) -> Union[VecNormalize, None]:
    """
    Return the `VecNormalize` wrapper if the current environment is wrapped in it.

    Args:
        venv (object): Environment in which

    Returns:
        Union[VecNormalize, None]
    """
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


def get_render_func(venv: gym.Env):
    """
    Get render function for the environment.

    Args:
        venv (object): Environment for which to retrieve the render function.

    Returns:
        Callable
    """
    print(venv)
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def make_env_thunk(env_name: str, env_configs: dict, seed: int, rank: int) -> Callable:
    """
    Returns a callable to create environments based on the specs provided.

    Args:
        env_name (str): Environment to create.
        env_configs (dict): Key word arguments for making the environment.
        seed (int): Random seed for the experiments.
        rank (int): "Rank" of the environment that the callable would return.

    Returns:
        Callable
    """

    def _thunk():
        env = gym.make(env_name, **env_configs)

        if not callable(getattr(env, "seed", None)):
            raise NotImplementedError(
                f"`seed` required for experiment replicability, but not implemented."
            )

        env.seed(seed + rank)

        if len(env.observation_space.shape) != 1:
            raise NotImplementedError

        env = RLSquaredEnv(env)

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            raise NotImplementedError

        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    env_kwargs: dict,
    seed: int,
    num_processes: int,
    device: torch.device,
    gamma: float,
    norm_observations: bool,
    norm_rewards: bool,
) -> PyTorchVecEnvWrapper:
    """
    Returns PyTorch compatible vectorized environments.

    Args:
        env_name (str): Name of the environment to be created.
        env_kwargs (dict): Key word arguments to create the environment.
        seed (int): Random seed for environments.
        num_processes (int): Number of parallel processes to be used for simulations.
        device (torch.device): Device to use with PyTorch tensors.
        gamma (float): Discount factor for the environment.
        norm_observations (bool): Whether to normalize envinronment observations.
        norm_rewards (bool): Whether to normalize envinronment rewards.

    Returns:
        PyTorchVecEnvWrapper
    """
    envs = [
        make_env_thunk(env_name, env_kwargs, seed, rank)
        for rank in range(num_processes)
    ]

    envs = MultiprocessingVecEnv(envs)

    if norm_observations or norm_rewards:
        envs = VecNormalize(
            envs, gamma=gamma, norm_obs=norm_observations, norm_reward=norm_rewards
        )

    envs = PyTorchVecEnvWrapper(envs, device)

    return envs


def register_custom_envs() -> None:
    """
    Register custom environments for experiments.

    Returns:
        None
    """
    register(
        id="PointNavigation-v1",
        entry_point="autometa.envs.point_mass.point_navigation_env:PointNavigationEnv",
    )

    register(
        id="CheetahVelocity-v1",
        entry_point="autometa.envs.cheetah.cheetah_velocity_env:CheetahVelocityEnv",
    )

    register(
        id="AntNavigation-v1",
        entry_point="autometa.envs.ant.ant_navigation_env:AntNavigationEnv",
    )

    register(
        id="AntVelocity-v1",
        entry_point="autometa.envs.ant.ant_velocity_env:AntVelocityEnv",
    )
    pass
