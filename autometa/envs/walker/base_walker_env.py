import numpy as np
from typing import Tuple, Any
from abc import ABC

import gym
from gym.envs.mujoco import Walker2dEnv as Walker2dEnv_

from autometa.envs.base_randomized_mujoco_env import BaseRandomizedMujocoEnv


class BaseWalkerEnv(Walker2dEnv_, BaseRandomizedMujocoEnv, ABC):
    def __init__(self, seed: int = None):
        """
        Initialize the Mujoco Ant environment for meta-learning.

        Args:
            seed (int): Random seed.
        """
        BaseRandomizedMujocoEnv.__init__(self, seed)
        Walker2dEnv_.__init__(self)
        pass

    def _get_obs(self) -> np.ndarray:
        """
        Get observation.

        Returns:
            np.ndarray
        """
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def get_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Returns the observation space and the action space.

        Returns:
            Tuple
        """
        return self.observation_space, self.action_space

    @property
    def observation_space(self) -> gym.Space:
        """
        Returns the observation space for the environment.

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
        Set the action space for the environment.

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

    def viewer_setup(self) -> None:
        """
        Set up the viewer for rendering the environment.

        Returns:
            None
        """
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
        pass

    def render(self, mode: str = "human"):
        """
        Render the enevironment.

        Args:
            mode (str): Mode in which to render the environment.

        Returns:
            None
        """
        if mode == "human":
            self._get_viewer(mode).render()
        else:
            raise NotImplementedError(f"`render` not implemented for `{mode}` mode.")
