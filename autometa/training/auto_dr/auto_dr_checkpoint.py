import torch

from autometa.training.base_training_checkpoint import BaseTrainingCheckpoint


class AutoDRCheckpoint(BaseTrainingCheckpoint):
    @classmethod
    def load(cls, absolute_path: str, device: torch.device) -> "AutoDRCheckpoint":
        """
        Load checkpoint from a given

        Args:
            absolute_path (str): Absolute path for loading the checkpoint.
            device (torch.device): Device specification to remap storage locations.
        """
        checkpoint_state = torch.load(absolute_path, map_location=device)

        return cls(
            wandb_run_id=checkpoint_state["wandb_run_id"],
            current_iteration=checkpoint_state["current_iteration"],
            actor_state_dict=checkpoint_state["actor_state_dict"],
            critic_state_dict=checkpoint_state["critic_state_dict"],
            optimizer_state_dict=checkpoint_state["optimizer_state_dict"],
            observations_rms=checkpoint_state["observations_rms"],
            rewards_rms=checkpoint_state["rewards_rms"],
            randomized_parameters=checkpoint_state["randomized_parameters"],
            randomization_buffer=checkpoint_state["randomization_buffer"],
        )

    @property
    def checkpoint_data(self) -> dict:
        """
        Return data to be saved for the checkpoint.

        Returns:
            dict
        """
        return {
            "wandb_run_id": self.wandb_run_id,
            "current_iteration": self.current_iteration,
            "actor_state_dict": self.actor_state_dict,
            "critic_state_dict": self.critic_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "observations_rms": self.observations_rms,
            "rewards_rms": self.rewards_rms,
            "randomized_parameters": self.randomized_parameters,
            "randomization_buffer": self.randomization_buffer,
        }
