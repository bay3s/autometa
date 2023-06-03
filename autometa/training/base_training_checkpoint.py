from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import os
import json
import pathlib

import torch

from stable_baselines3.common.running_mean_std import RunningMeanStd

from autometa.randomization.randomization_performance_buffer import (
    RandomizationPerformanceBuffer,
)
from autometa.randomization.randomization_parameter import RandomizationParameter


@dataclass
class BaseTrainingCheckpoint(ABC):
    """
    Base dataclass to keep track of experiment configs.

    Params:
        current_iteration (int): Current policy iteration.
        actor_state_dict (str): State of the actor.
        critic_state_dict (dict): State of the critic.
        optimizer_state_dict (dict): State of the optimizer.
        observation_rms (RunningMeanStd): `RunningMeanStd` for the environment observations.
        rewards_rms (RunningMeanStd): `RunningMeanStd` for the environment rewards.
        wandb_run_id (str): `wandb` run id.
        randomization_parameters (List[RandomizationParameter]): A list of randomized parameters.
        randomization_buffer (RandomizationPerformanceBuffer): Performance buffer for Auto-DR.
    """

    # logs
    wandb_run_id: str
    current_iteration: int

    # actor-critic / ppo
    actor_state_dict: dict
    critic_state_dict: dict
    optimizer_state_dict: dict

    # rms
    observations_rms: RunningMeanStd
    rewards_rms: RunningMeanStd

    # adr
    randomized_parameters: List[RandomizationParameter] = None
    randomization_buffer: RandomizationPerformanceBuffer = None
    pass

    @classmethod
    @abstractmethod
    def load(cls, absolute_path: str, device: torch.device) -> "BaseTrainingCheckpoint":
        """
        Load checkpoint from a given

        Args:
            absolute_path (str): Absolute path for loading the checkpoint.
            device (torch.device): Device specification to remap storage locations.
        """
        raise NotImplementedError

    @property
    def json(self) -> str:
        """
        Return JSON string with dataclass fields.

        Returns:
            str
        """
        return json.dumps(self.__dict__, indent=2)

    @property
    @abstractmethod
    def checkpoint_data(self) -> dict:
        """
        Return data to be saved for the checkpoint.

        Returns:
            dict
        """
        raise NotImplementedError

    def save(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        Returns the checkpoint directory.

        Args:
            checkpoint_dir (str): Absolute path for the directory in which to save the checkpoint.
            checkpoint_name (str): Name of the checkpoint.

        Returns:
            str
        """
        if not os.path.exists(checkpoint_dir):
            pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if checkpoint_name:
            checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}.pt"
        else:
            checkpoint_path = f"{checkpoint_dir}/checkpoint.pt"

        torch.save(self.checkpoint_data, checkpoint_path)
        pass
