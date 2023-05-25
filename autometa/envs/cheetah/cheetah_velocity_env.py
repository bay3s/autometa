from typing import Tuple, Optional

import numpy as np
import gym
from gym.utils import EzPickle

from autometa.envs.cheetah.base_cheetah_env import BaseCheetahEnv

from autometa.randomization.randomization_parameter import RandomizationParameter
from autometa.randomization.randomization_bound_type import RandomizationBoundType
from autometa.randomization.randomization_bound import RandomizationBound


class CheetahVelocityEnv(BaseCheetahEnv, EzPickle):

    RANDOMIZABLE_PARAMETERS = []

    def __init__(
        self,
        episode_length: int = 100,
        min_velocity: float = 0.0,
        max_velocity: float = 3.0,
        auto_reset: bool = True,
        seed: int = None,
    ):
        """
        Half-Cheetah environment with target velocity, as described in [1].

        The code is adapted from https://github.com/cbfinn/maml_rl

        The half-cheetah follows the dynamics from MuJoCo [2], and receives at each time step a reward composed of a
        control cost and a penalty equal to the difference between its current velocity and the target velocity.

        The tasks are generated by sampling the target velocities from the uniform distribution on [0, 2].

        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
            Networks", 2017 (https://arxiv.org/abs/1703.03400)
        [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for model-based control", 2012
            (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

        Args:
            episode_length (int): Maximum number of steps per episode.
            min_velocity (float): Minimum target velocity.
            max_velocity (float): Maximum target velocity.
            seed (int): Random seed.
        """
        self._episode_length = episode_length
        self._elapsed_steps = 0
        self._auto_reset = auto_reset
        self._episode_reward = 0.0

        self._min_velocity = min_velocity
        self._max_velocity = max_velocity

        # set a stub, sample later.
        self._target_velocity = np.random.uniform(
            self._min_velocity, self._max_velocity, size=1
        )

        BaseCheetahEnv.__init__(self)
        EzPickle.__init__(self)

        # sample
        self.seed(seed)
        self.sample_task()
        pass

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the action space

        Returns:
            Tuple[gym.Space, gym.Space]
        """
        return self.observation_space, self.action_space

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
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        self._episode_reward += reward

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

    def sample_task(self, task: dict = None):
        """
        Sample a new target velocity.

        Returns:
            None
        """
        # @todo update this.
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
