from typing import Tuple, Any, List, Optional
import numpy as np

import gym
from gym.utils import EzPickle

from autometa.envs.ant.base_ant_env import BaseAntEnv
from autometa.randomization.randomization_parameter import RandomizationParameter
from autometa.randomization.randomization_bound_type import RandomizationBoundType
from autometa.randomization.randomization_bound import RandomizationBound


class AntNavigationEnv(BaseAntEnv, EzPickle):
    RANDOMIZABLE_PARAMETERS = [
        RandomizationParameter(
            name="x_position",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-3.0,
                max_value=0.0,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=3.0,
            ),
            delta=0.05,
        ),
        RandomizationParameter(
            name="y_position",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0.0,
                min_value=-3.0,
                max_value=0.0,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0.0,
                min_value=0.0,
                max_value=3.0,
            ),
            delta=0.05,
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
        Ant environment with rewards based on achieving a target position for Mujoco Ant locomotor.

        The ant follows the dynamics from MuJoCo [1], and receives at each time step a reward composed of a control
        cost, a contact cost, a survival reward, and a penalty equal to its L1 distance to the target position. The
        tasks are generated by sampling the target positions from the uniform
        distribution on [-3, 3]^2.

        [1] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

        Args:
            episode_length (int): Max number of episode steps.
            randomizable_parameters (List[RandomizationParameter]): List of randomized parameters and specs.
            auto_reset (bool): Auto-reset the environment after completion.
            seed (int): Random seed for sampling.
        """
        self._episode_length = episode_length
        self._elapsed_steps = 0
        self._auto_reset = auto_reset
        self._episode_reward = 0.0

        # stub
        self._current_state = None
        self._target_state = None

        # params
        self._randomized_parameters = self._init_params(randomizable_parameters)

        BaseAntEnv.__init__(self)
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

    def sample_task(self, task: dict = None) -> None:
        """
        Sample a new goal position for the navigation task

        Returns:
            None
        """
        if task is None:
            task = dict()

            x_param = self.randomized_parameter("x_position")
            task["x_position"] = self.np_random.uniform(
                x_param.lower_bound.min_value, x_param.upper_bound.max_value
            )

            y_param = self.randomized_parameter("y_position")
            task["y_position"] = self.np_random.uniform(
                y_param.lower_bound.min_value, y_param.upper_bound.max_value
            )
            pass

        self._target_state = np.concatenate(
            [
                [task["x_position"]],
                [task["y_position"]],
            ],
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

        return BaseAntEnv.reset(self, seed=seed, options=options)

    def step(self, action: np.ndarray) -> Tuple:
        """
        Take a step in the environment and return the corresponding observation, action, reward, plus additional info.

        Args:
            action (np.ndarray): Action to be taken in the environment.

        Returns:
            Tuple
        """
        self._elapsed_steps += 1

        self.do_simulation(action, self.frame_skip)
        new_position = self.get_body_com("torso")[:2]

        goal_reward = -1.0 * np.abs(new_position - self._target_state).sum()
        survive_reward = 0.0

        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = (
            0.5 * 1e-3 * np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)).sum()
        )

        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        self._episode_reward += reward

        observation = self._get_obs()
        terminated = False
        truncated = self.elapsed_steps == self.max_episode_steps
        done = truncated or terminated

        info = {}
        if done:
            info["episode"] = {}
            info["episode"]["r"] = self._episode_reward

            if self._auto_reset:
                observation, _ = self.reset()
                pass

        return observation, reward, terminated, truncated, info

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space of the environment.

        Returns:
          gym.Space
        """
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: Any) -> None:
        """
        Set the observation space for the environment.

        Returns:
          gym.Space
        """
        self._observation_space = value

    @property
    def action_space(self) -> gym.Space:
        """
        Returns the action space

        Returns:
          gym.Space
        """
        return self._action_space

    @action_space.setter
    def action_space(self, value: Any) -> None:
        """
        Set the action space for the environment.

        Returns:
            gym.Space
        """
        self._action_space = value

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Get the observation space and the action space.

        Returns:
            Tuple
        """
        return self._observation_space, self._action_space

    @property
    def elapsed_steps(self) -> int:
        """
        Returns the elapsed number of episode steps in the environment.

        Returns:
          int
        """
        return self._elapsed_steps

    @property
    def max_episode_steps(self) -> int:
        """
        Returns the maximum number of episode steps in the environment.

        Returns:
          int
        """
        return self._episode_length

    def close(self) -> None:
        """
        Close the environment.

        Returns:
          None
        """
        pass
