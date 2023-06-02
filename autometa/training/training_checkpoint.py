from dataclasses import dataclass
import os
import json
import pathlib

import torch

from stable_baselines3.common.running_mean_std import RunningMeanStd


@dataclass
class TrainingCheckpoint:
    """
    Base dataclass to keep track of experiment configs.

    Params:
      current_iteration (int): Current policy iteration.
      actor_state_dict (str): State of the actor.
      critic_state_dict (dict): State of the critic.
      optimizer_state_dict (dict): State of the optimizer.
      observation_rms (RunningMeanStd): `RunningMeanStd` for the environment observations.
      rewards_rms (RunningMeanStd): `RunningMeanStd` for the environment rewards.
      training_config (TrainingConfig): Training configs for the experiment.
    """

    current_iteration: int
    actor_state_dict: dict
    critic_state_dict: dict
    optimizer_state_dict: dict
    observations_rms: RunningMeanStd
    rewards_rms: RunningMeanStd
    pass

    @classmethod
    def load(cls, absolute_path: str, device: torch.device) -> "TrainingCheckpoint":
        """
        Load checkpoint from a given

        Args:
            absolute_path (str): Absolute path for loading the checkpoint.
            device (torch.device): Device specification to remap storage locations.
        """
        checkpoint_state = torch.load(absolute_path, map_location=device)

        return cls(
            current_iteration=checkpoint_state["current_iteration"],
            actor_state_dict=checkpoint_state["actor_state_dict"],
            critic_state_dict=checkpoint_state["critic_state_dict"],
            optimizer_state_dict=checkpoint_state["optimizer_state_dict"],
            observations_rms=checkpoint_state["observations_rms"],
            rewards_rms=checkpoint_state["rewards_rms"],
        )

    @property
    def json(self) -> str:
        """
        Return JSON string with dataclass fields.

        Returns:
            str
        """
        return json.dumps(self.__dict__, indent=2)

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
            checkpoint_path = f"{checkpoint_dir}/checkpoint-{checkpoint_name}.pt"
        else:
            checkpoint_path = f"{checkpoint_dir}/checkpoint.pt"

        # data
        checkpoint_data = {
            "current_iteration": self.current_iteration,
            "actor_state_dict": self.actor_state_dict,
            "critic_state_dict": self.critic_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "observations_rms": self.observations_rms,
            "rewards_rms": self.rewards_rms,
        }

        # save
        torch.save(checkpoint_data, checkpoint_path)
        pass
