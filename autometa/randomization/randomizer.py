from typing import List, Tuple

import random

import gym
import numpy as np

from autometa.randomization.randomization_performance_buffer import (
    RandomizationPerformanceBuffer,
)
from autometa.randomization.randomization_parameter import RandomizationParameter
from autometa.randomization.randomization_bound_type import RandomizationBoundType
from autometa.randomization.randomization_boundary import RandomizationBoundary
from autometa.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper


class Randomizer:
    def __init__(
        self,
        parallel_envs: PyTorchVecEnvWrapper,
        evaluation_probability: float,
        buffer_size: int,
        deltas: dict,
        performance_threshold_lower: float,
        performance_threshold_upper: float,
    ) -> None:
        """
        Automatic Domain Randomization (ADR) based on the Open AI paper.

        Args:
            parallel_envs (int): Number of environments being randomized in parallel.
            evaluation_probability (float): Probability of boundary sampling and subsequently increasing the difficulty.
            buffer_size (int): Minimum buffer size required for evaluating boundary sampling performance.
            deltas (dict): Mapping of individual parameters to their individual deltas.
            performance_threshold_upper (float): Lower threshold for performance on a specific environment, if this is
                not met then the parameter entropy is decreased.
            performance_threshold_lower (float): Lower threshold for performance on a specific environment, if this is
                met then the parameter entropy is increased.
        """
        self.parallel_envs = parallel_envs

        randomizable_params = self.parallel_envs.env_method(
            "randomizable_parameters", indices=0
        )[0]

        self.randomized_parameters = self._init_params(randomizable_params, deltas)
        self.buffer = RandomizationPerformanceBuffer(
            randomizable_params, buffer_size=buffer_size
        )

        self.evaluation_probability = evaluation_probability
        self.buffer_size = buffer_size
        self.sampled_boundaries = [None] * parallel_envs.num_envs

        # performance
        self.lower_performance_threshold = performance_threshold_lower
        self.upper_performance_threshold = performance_threshold_upper
        pass

    @staticmethod
    def _init_params(params: List[RandomizationParameter], deltas: dict) -> dict:
        """
        Convert a list of parameters to dict.

        Args:
            params (List[RandomizationParameter]): A list of randomized parameters.
            deltas (dict): Mapping of individual parameters to their individual deltas.

        Returns:
            dict
        """
        randomized = dict()

        for param in params:
            param.delta = deltas[param.name]
            randomized[param.name] = param

        return randomized

    def entropy(self, eps: float = 1e-12) -> float:
        """
        Evaluate the current entropy of the parameters.

        Args:
          eps (float): Stub to avoid numerical issues.

        Returns:
          float
        """
        ranges = list()

        for param in self.randomized_parameters.values():
            ranges.append(param.range + eps)

        return np.log(ranges).mean()

    def _re_evaluate_boundary(self, sampled_boundary: RandomizationBoundary) -> None:
        """
        Update ADR bounds based on the performance for a given boundary.

        Args:
          sampled_boundary (RandomizationBoundary): Sampled boundary to evaluate.

        Returns:
          None
        """
        if not self.buffer.is_full(sampled_boundary):
            return

        performance = np.mean(np.array(self.buffer.get(sampled_boundary)))
        self.buffer.truncate(sampled_boundary)

        param = sampled_boundary.parameter
        bound = sampled_boundary.bound

        # increase entropy
        if performance >= self.upper_performance_threshold:
            if bound.type == RandomizationBoundType.UPPER_BOUND:
                self.randomized_parameters[param.name].increase_upper_bound()
            elif bound.type == RandomizationBoundType.LOWER_BOUND:
                self.randomized_parameters[param.name].decrease_lower_bound()
            else:
                raise ValueError

        # decrease entropy
        if performance < self.lower_performance_threshold:
            if bound.type == RandomizationBoundType.UPPER_BOUND:
                self.randomized_parameters[param.name].decrease_upper_bound()
            elif bound.type == RandomizationBoundType.LOWER_BOUND:
                self.randomized_parameters[param.name].increase_lower_bound()
            else:
                raise ValueError

    def re_evaluate(self) -> None:
        """
        Update ADR bounds based on the performance.

        Returns:
            None
        """
        for boundary in self.sampled_boundaries:
            if boundary is None:
                continue

            # evaluate
            self._re_evaluate_boundary(boundary)
            pass

    def _get_task(self) -> Tuple:
        """
        Get randomized parameter values.

        Returns:
          Tuple
        """
        randomized_params = dict()

        # boundary
        sampled_boundary = None

        for param in self.randomized_parameters.values():
            lower_bound = param.lower_bound
            upper_bound = param.upper_bound

            randomized_params[param.name] = np.random.uniform(
                lower_bound.value, upper_bound.value
            )

        # adr
        if np.random.uniform(0, 1) <= self.evaluation_probability:
            # param
            sampled_param = random.choice(
                [
                    param
                    for param in self.randomized_parameters.values()
                    if not param.frozen
                ]
            )

            # bound
            param_bounds = list([sampled_param.lower_bound, sampled_param.upper_bound])
            sampled_bound = random.choice(
                [bound for bound in param_bounds if not bound.frozen]
            )

            # boundary sampling
            if sampled_bound.type == RandomizationBoundType.UPPER_BOUND:
                randomized_params[sampled_param.name] = sampled_bound.value
            elif sampled_bound.type == RandomizationBoundType.LOWER_BOUND:
                randomized_params[sampled_param.name] = sampled_bound.value
            else:
                raise ValueError

            sampled_boundary = RandomizationBoundary(
                parameter=sampled_param, bound=sampled_bound
            )
            pass

        return randomized_params, sampled_boundary

    def randomize_all(self) -> None:
        """
        Sample tasks for each environment.

        Returns:
          None
        """
        zipped = zip(range(self.parallel_envs.num_envs), self.sampled_boundaries)
        new_tasks = list()

        for env_idx, boundary in zipped:
            randomized_params, boundary = self._get_task()
            self.sampled_boundaries[env_idx] = boundary
            new_tasks.append(randomized_params)

        self.parallel_envs.sample_tasks_async(np.array(new_tasks))
        pass

    def update_buffer(
        self, sampled_boundary: RandomizationBoundary, episode_return: float
    ) -> None:
        """
        Update buffer with the sampled boundary and associated episode return.

        Args:
          sampled_boundary (RandomizationBoundary): Parameter boundary sampled for Auto DR.
          episode_return (float): Episode return for the sampled boundary.

        Returns:
          None
        """
        self.buffer.insert(sampled_boundary, episode_return)

    def _on_step(self, dones: List, infos: List) -> None:
        """
        Randomizer logic to be executed after each environment step.

        - Update the performance buffer with the episode returns.
        - Update randomization bounds and entropy.
        - Propagate updates to the environment.

        Args:
          dones (List): List of boolean values indicating whether an episode is done.
          infos (List): Info related to the current environment step.

        Returns:
          None
        """
        zipped = zip(
            dones, infos, range(self.parallel_envs.num_envs), self.sampled_boundaries
        )

        for done, info, env_idx, boundary in zipped:
            if not done:
                continue

            if boundary is not None and "meta_episode" in info.keys():
                self.update_buffer(boundary, info["meta_episode"]["r"])

    @property
    def observation_space(self) -> gym.Space:
        """
        Return the observation space for the environment.

        Returns:
            gym.Space
        """
        return self.parallel_envs.observation_space

    @property
    def action_space(self) -> gym.Space:
        """
        Return the action space for the environment.

        Returns:
            gym.Space
        """
        return self.parallel_envs.action_space

    @property
    def num_envs(self) -> int:
        """
        Return the number of parallel environments that the Randomizer is controlling.

        Returns:
            int
        """
        return self.parallel_envs.num_envs

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Take a step in the environments and return a tuple of observations, rewards, dones, infos.

        Args:
            actions (np.ndarray): Actions to take in the randomized environments.

        Returns:
            Tuple
        """
        obs, rewards, dones, infos = self.parallel_envs.step(actions)
        self._on_step(dones, infos)

        return obs, rewards, dones, infos

    @property
    def info(self) -> dict:
        """
        Returns info regarding a specific randomizer instance.

        Returns:
            dict
        """
        info = dict()

        for param in self.randomized_parameters.values():
            # boundaries
            lower_boundary = RandomizationBoundary(param, param.lower_bound)
            upper_boundary = RandomizationBoundary(param, param.upper_bound)

            # buffers
            lower_buffer = self.buffer.get(lower_boundary)
            upper_buffer = self.buffer.get(upper_boundary)

            # bounds
            info[f"randomizer/bound/{param.name}_upper"] = param.upper_bound.value
            info[f"randomizer/bound/{param.name}_lower"] = param.lower_bound.value

            # buffer
            info[f"randomizer/buffer_size/{param.name}_upper"] = len(upper_buffer)
            info[f"randomizer/buffer_size/{param.name}_lower"] = len(lower_buffer)

            # lower
            info[f"randomizer/rewards/{param.name}_upper"] = (
                np.mean(upper_buffer) if len(upper_buffer) else 0.0
            )
            info[f"randomizer/rewards/{param.name}_lower"] = (
                np.mean(lower_buffer) if len(lower_buffer) else 0.0
            )

            # range
            info[f"randomizer/range/{param.name}"] = param.range
            pass

        info["randomizer/entropy"] = self.entropy()

        return info
