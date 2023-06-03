import numpy as np
from typing import Tuple, Any
from abc import ABC

import gym
from gym.spaces import Box
from gym.envs.mujoco import AntEnv as AntEnv_

from autometa.envs.base_randomized_mujoco_env import BaseRandomizedMujocoEnv


class BaseAntEnv(AntEnv_, BaseRandomizedMujocoEnv, ABC):
    def __init__(self, seed: int = None):
        """
        Initialize the Mujoco Ant environment for meta-learning.

        Args:
            seed (int): Random seed.
        """
        BaseRandomizedMujocoEnv.__init__(self, seed)
        AntEnv_.__init__(self)

        # overwrite
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(125,), dtype=np.float32
        )

        self._action_scaling = None
        pass

    def _get_obs(self) -> np.ndarray:
        """
        Get observation.

        Returns:
            np.ndarray
        """
        return (
            np.concatenate(
                [
                    self.sim.data.qpos.flat,
                    self.sim.data.qvel.flat,
                    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                    self.sim.data.get_body_xmat("torso").flat,
                    self.get_body_com("torso").flat,
                ]
            )
            .astype(np.float32)
            .flatten()
        )

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
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True

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

    def viewer_setup(self) -> None:
        """
        Set up the viewer for rendering the environment.

        Returns:
            None
        """
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True

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
