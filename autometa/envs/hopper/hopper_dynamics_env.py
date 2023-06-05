from typing import Tuple, Optional, List

import numpy as np
from gym.utils import EzPickle, seeding

from autometa.envs.hopper.base_hopper_env import BaseHopperEnv
from autometa.randomization.randomization_parameter import RandomizationParameter
from autometa.randomization.randomization_bound_type import RandomizationBoundType
from autometa.randomization.randomization_bound import RandomizationBound


class HopperDynamicsEnv(BaseHopperEnv, EzPickle):
    THIGH_MASS_INIT = 3.92699082
    THIGH_MASS_IDX = 2

    LEG_MASS_INIT = 2.71433605
    LEG_MASS_IDX = 3

    FOOT_MASS_INIT = 5.0893801
    FOOT_MASS_IDX = 4

    RANDOMIZABLE_PARAMETERS = [
        RandomizationParameter(
            name="thigh_mass",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=THIGH_MASS_INIT,
                min_value=THIGH_MASS_INIT - 2.0,
                max_value=THIGH_MASS_INIT,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=THIGH_MASS_INIT,
                min_value=THIGH_MASS_INIT,
                max_value=THIGH_MASS_INIT + 2.0,
            ),
            delta=0.05,
        ),
        RandomizationParameter(
            name="leg_mass",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=LEG_MASS_INIT,
                min_value=LEG_MASS_INIT - 2.0,
                max_value=LEG_MASS_INIT,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=LEG_MASS_INIT,
                min_value=LEG_MASS_INIT,
                max_value=LEG_MASS_INIT + 2.0,
            ),
            delta=0.05,
        ),
        RandomizationParameter(
            name="foot_mass",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=FOOT_MASS_INIT,
                min_value=FOOT_MASS_INIT - 2.0,
                max_value=FOOT_MASS_INIT,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=FOOT_MASS_INIT,
                min_value=FOOT_MASS_INIT,
                max_value=FOOT_MASS_INIT + 2.0,
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
        Half-Cheetah environment with target velocity, as described in [1].

        The half-cheetah follows the dynamics from MuJoCo [2], and receives at each time step a reward composed of a
        control cost and a penalty equal to the difference between its current velocity and the target velocity.

        The tasks are generated by sampling the target velocities from the uniform distribution on [0, 2].

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

        # params
        self._randomized_parameters = self._init_params(randomizable_parameters)

        BaseHopperEnv.__init__(self)
        EzPickle.__init__(self)

        # @todo remove
        self.log_scale_limit = 3.0
        self.init_params = dict()
        self.init_params['body_mass'] = self.model.body_mass
        self.init_params['body_inertia'] = self.model.body_inertia
        self.init_params['dof_damping'] = self.model.dof_damping
        self.init_params['geom_friction'] = self.model.geom_friction
        self.cur_params = self.init_params

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

        observation, reward, terminated, truncated, info = BaseHopperEnv.step(
            self, action
        )

        self._episode_reward += reward
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

    def sample_task(self, task: dict = None):
        """
        Sample a new target velocity.

        Args:
            task (dict): Task that has been sampled, defaults to none.

        Returns:
            None
        """
        new_params = {}

        body_mass_multipliers = np.array(1.5) ** np.random.uniform(
            -self.log_scale_limit, self.log_scale_limit, size=self.model.body_mass.shape
        )

        new_params["body_mass"] = self.init_params["body_mass"] * body_mass_multipliers

        # body_inertia
        body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(
            -self.log_scale_limit,
            self.log_scale_limit,
            size=self.model.body_inertia.shape,
        )

        new_params["body_inertia"] = body_inertia_multiplyers * self.init_params["body_inertia"]

        dof_damping_multipliers = np.array(1.3) ** np.random.uniform(
            -self.log_scale_limit,
            self.log_scale_limit,
            size=self.model.dof_damping.shape,
        )

        new_params["dof_damping"] = np.multiply(self.init_params["dof_damping"], dof_damping_multipliers)

        # friction at the body components
        dof_damping_multipliers = np.array(1.5) ** np.random.uniform(
            -self.log_scale_limit,
            self.log_scale_limit,
            size=self.model.geom_friction.shape,
        )
        new_params["geom_friction"] = np.multiply(
            self.init_params["geom_friction"], dof_damping_multipliers
        )

        self.set_task(new_params)
        pass

    def set_task(self, task: dict) -> None:
        """
        Update the dynamics of the hopper with the task.

        Returns:
            None
        """
        self.model.body_mass[:] = task["body_mass"][:]
        self.model.body_inertia[:] = task["body_inertia"][:]
        self.model.dof_damping[:] = task["dof_damping"][:]
        self.model.geom_friction[:] = task["geom_friction"][:]
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
        if seed is not None:
            self.np_random, seed = seeding.np_random(seed)

        self._elapsed_steps = 0
        self._episode_reward = 0.0

        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        return BaseHopperEnv.reset(self, seed=seed, options=options)

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
