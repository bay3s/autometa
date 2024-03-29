from typing import Tuple, Optional, List

import numpy as np
from gym.utils import EzPickle

from autometa.envs.cheetah.base_cheetah_env import BaseCheetahEnv

from autometa.randomization.randomization_parameter import RandomizationParameter
from autometa.randomization.randomization_bound_type import RandomizationBoundType
from autometa.randomization.randomization_bound import RandomizationBound


class CheetahVelocityEnv(BaseCheetahEnv, EzPickle):
    RANDOMIZABLE_PARAMETERS = [
        RandomizationParameter(
            name="target_velocity",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0.05,
                min_value=0.05,
                max_value=0.05,
                frozen=True,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0.05,
                min_value=0.05,
                max_value=3.0,
            ),
            delta=0.03,
        ),
    ]

    def __init__(
        self,
        episode_length: int,
        randomizable_parameters: List[RandomizationParameter] = RANDOMIZABLE_PARAMETERS,
        auto_reset: bool = True,
        seed: int = None,
    ):
        """
        Half-Cheetah environment with target velocity, as described in [1].

        The half-cheetah follows the dynamics from MuJoCo [2], and receives at each time step a reward composed of a
        control cost and a penalty equal to the difference between its current velocity and the target velocity.

        The tasks are generated by sampling the target velocities from the uniform distribution on [0, 3].

        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
            Networks", 2017 (https://arxiv.org/abs/1703.03400)
        [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for model-based control", 2012
            (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

        Args:
            episode_length (int): Maximum number of steps per episode.
            randomizable_parameters (List[RandomizableParamter]): Randomizable parameters.
            auto_reset (bool): Whether to auto-reset at the end of episode.
            seed (int): Random seed.
        """
        self._episode_length = episode_length
        self._elapsed_steps = 0
        self._auto_reset = auto_reset
        self._episode_reward = 0.0

        # stub
        self._target_velocity = None

        # params
        self._randomized_parameters = self._init_params(randomizable_parameters)

        BaseCheetahEnv.__init__(self)
        EzPickle.__init__(self)

        # sample
        self.seed(seed)
        self.sample_task()
        pass

    @staticmethod
    def _init_params(params: List[RandomizationParameter]) -> dict:
        """
        Convert a list of parameters to dict.

        Args:
            params (List[RandomizationParameter]): A list of randomized parameters.

        Returns:
            dict
        """
        randomized = dict()
        for param in params:
            randomized[param.name] = param

        return randomized

    def randomizable_parameters(self) -> List[RandomizationParameter]:
        """
        Return a list of randomized parameters.

        Returns:
            List[RandomizedParameter]
        """
        return self.RANDOMIZABLE_PARAMETERS

    def randomized_parameter(self, param_name: str) -> RandomizationParameter:
        """
        Return the randomized parameter.

        Args:
            param_name (str): Name of the parameter to return.

        Returns:
            RandomizationParameter
        """
        return self._randomized_parameters[param_name]

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward,
        additional info, etc.

        Args:
            action (np.ndarray): Action to be taken in the environment.

        Returns:
            Tuple
        """
        self._elapsed_steps += 1

        position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        position_after = self.sim.data.qpos[0]

        forward_vel = (position_after - position_before) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._target_velocity.item())
        ctrl_cost = 0.5 * 1e-1 * np.square(action).sum()

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        self._episode_reward += reward

        terminated = False
        truncated = self.elapsed_steps == self.max_episode_steps
        done = truncated or terminated

        info = dict()
        info["sampled_task"] = self._target_velocity.item()

        if done:
            info["episode"] = {}
            info["episode"]["r"] = self._episode_reward

            if self._auto_reset:
                observation, _ = self.reset()
                pass

        return observation, reward, terminated, truncated, info

    def sample_task(self, task: dict = None):
        """
        Sample a new target velocity.

        Returns:
            None
        """
        if task is None:
            task = dict()

            target_velocity = self.randomized_parameter("target_velocity")
            task["target_velocity"] = self.np_random.uniform(
                target_velocity.lower_bound.min_value,
                target_velocity.upper_bound.max_value,
            )
            pass

        self._target_velocity = np.concatenate(
            [[task["target_velocity"]]],
            dtype=np.float32,
        )
        pass

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple:
        """
        Reset the environment to the start state.

        Args:
            seed (int): Random seed.
            options (dict): Additional options.

        Returns:
            Tuple
        """
        self._elapsed_steps = 0
        self._episode_reward = 0.0

        return BaseCheetahEnv.reset(self, seed=seed, options=options)

    @property
    def elapsed_steps(self) -> int:
        """
        Return the elapsed steps.

        Returns:
            int
        """
        return self._elapsed_steps

    @property
    def max_episode_steps(self) -> int:
        """
        Return the maximum episode steps.

        Returns:
            int
        """
        return self._episode_length
