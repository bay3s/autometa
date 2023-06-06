from typing import Tuple, Optional, List

from copy import deepcopy
import numpy as np
from gym.utils import EzPickle, seeding

from autometa.envs.walker.base_walker_env import BaseWalkerEnv
from autometa.randomization.randomization_parameter import RandomizationParameter
from autometa.randomization.randomization_bound_type import RandomizationBoundType
from autometa.randomization.randomization_bound import RandomizationBound


class WalkerDynamicsEnv(BaseWalkerEnv, EzPickle):
    SCALING_FACTOR = 3.0

    RANDOMIZABLE_PARAMETERS = [
        RandomizationParameter(
            name="mass_scaling",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0,
                min_value=-SCALING_FACTOR,
                max_value=0,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0,
                min_value=0,
                max_value=SCALING_FACTOR,
            ),
            delta=0.05,
        ),
        RandomizationParameter(
            name="inertia_scaling",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0,
                min_value=-SCALING_FACTOR,
                max_value=0,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0,
                min_value=0,
                max_value=SCALING_FACTOR,
            ),
            delta=0.05,
        ),
        RandomizationParameter(
            name="damping_scaling",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0,
                min_value=-SCALING_FACTOR,
                max_value=0,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0,
                min_value=0,
                max_value=SCALING_FACTOR,
            ),
            delta=0.05,
        ),
        RandomizationParameter(
            name="friction_scaling",
            lower_bound=RandomizationBound(
                type=RandomizationBoundType.LOWER_BOUND,
                value=0,
                min_value=-SCALING_FACTOR,
                max_value=0,
            ),
            upper_bound=RandomizationBound(
                type=RandomizationBoundType.UPPER_BOUND,
                value=0,
                min_value=0,
                max_value=SCALING_FACTOR,
            ),
            delta=0.05,
        ),
    ]

    MASS_COEFFICIENT = 1.5
    INERTIA_COEFFICIENT = 1.5
    DAMPING_COEFFICIENT = 1.3
    FRICTION_COEFFICIENT = 1.5

    def __init__(
        self,
        episode_length: int,
        randomizable_parameters: List[RandomizationParameter] = RANDOMIZABLE_PARAMETERS,
        auto_reset: bool = True,
        seed: int = None,
    ):
        """
        Wrapper to introduce randomization of physical dynamics in the Mujoco Walker environment.

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

        BaseWalkerEnv.__init__(self)
        EzPickle.__init__(self)

        # initial
        self._initial_damping = deepcopy(self.model.dof_damping)
        self._initial_mass = deepcopy(self.model.body_mass)
        self._initial_inertia = deepcopy(self.model.body_inertia)
        self._initial_friction = deepcopy(self.model.geom_friction)

        # randomization
        self._randomized_parameters = self._init_params(randomizable_parameters)

        # sample
        self.seed(seed)
        self.sample_task()
        pass

    @staticmethod
    def _init_params(params: List[RandomizationParameter]) -> dict:
        """
        Convert a list of parameters to dict.

        Args:
            params (List[RandomizationParameter]): A list of randomzed parameters.

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

        # position
        position_before = self.sim.data.qpos[0]

        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)

        position_after, height, ang = self.sim.data.qpos[0:3]

        # reward
        alive_bonus = 1.0
        reward = (position_after - position_before) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        self._episode_reward += reward

        # done
        terminated = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        truncated = self.elapsed_steps == self.max_episode_steps
        done = truncated or terminated

        # observation
        observation = self._get_obs()

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
        if task is None:
            task = dict()

            mass_scaling = self.randomized_parameter("mass_scaling")
            task["mass_scaling"] = self.np_random.uniform(
                mass_scaling.lower_bound.min_value,
                mass_scaling.upper_bound.max_value,
                size=self.model.body_mass.shape,
            )

            inertia_scaling = self.randomized_parameter("inertia_scaling")
            task["inertia_scaling"] = self.np_random.uniform(
                inertia_scaling.lower_bound.min_value,
                inertia_scaling.upper_bound.max_value,
                size=self.model.body_inertia.shape,
            )

            damping_scaling = self.randomized_parameter("damping_scaling")
            task["damping_scaling"] = self.np_random.uniform(
                damping_scaling.lower_bound.min_value,
                damping_scaling.upper_bound.max_value,
                size=self.model.dof_damping.shape,
            )

            friction_scaling = self.randomized_parameter("friction_scaling")
            task["friction_scaling"] = self.np_random.uniform(
                friction_scaling.lower_bound.min_value,
                friction_scaling.upper_bound.max_value,
                size=self.model.geom_friction.shape,
            )
            pass

        # update
        new_params = self._compute_sim_params(task)
        self._update_sim(new_params)
        pass

    def _compute_sim_params(self, task: dict) -> dict:
        """
        Compute updated simulation parameters based on the sampled task.

        Args:
            task (dict): Sampled task

        Returns:
            dict
        """
        mass_multiplier = np.array(self.MASS_COEFFICIENT) ** task["mass_scaling"]
        inertia_multiplier = (
            np.array(self.INERTIA_COEFFICIENT) ** task["inertia_scaling"]
        )
        damping_multiplier = (
            np.array(self.DAMPING_COEFFICIENT) ** task["damping_scaling"]
        )
        friction_multiplier = (
            np.array(self.FRICTION_COEFFICIENT) ** task["friction_scaling"]
        )

        return {
            "body_mass": self._initial_mass * mass_multiplier,
            "body_inertia": self._initial_inertia * inertia_multiplier,
            "dof_damping": self._initial_damping * damping_multiplier,
            "geom_friction": self._initial_friction * friction_multiplier,
        }

    def _update_sim(self, params: dict) -> None:
        """
        Update the simulation dynamics.

        Args:
            params (dict): New parameters for the simulation.

        Returns:
            None
        """
        self.model.body_mass[:] = params["body_mass"][:]
        self.model.body_inertia[:][:] = params["body_inertia"][:][:]
        self.model.dof_damping[:] = params["dof_damping"][:]
        self.model.geom_friction[:] = params["geom_friction"][:]
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

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )

        return self._get_obs(), {}

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
