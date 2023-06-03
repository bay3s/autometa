from abc import ABC, abstractmethod

import torch
import wandb

from autometa.training.base_training_config import BaseTrainingConfig
from autometa.learners.ppo import PPO

from autometa.utils.env_utils import make_vec_envs
from autometa.networks.stateful.stateful_actor_critic import StatefulActorCritic


class BaseTrainer(ABC):
    def __init__(self, config: BaseTrainingConfig = None, checkpoint_path: str = None):
        """
        Initialize an instance of a trainer for PPO.

        Args:
                config (BaseTrainingConfig): Params to be used for the trainer.
                checkpoint_path (str): Checkpoint path from which to restart.
        """
        self.config = config
        self.current_iteration = 0
        self.checkpoint = None
        self.wandb_initialized = False
        self.torch_device = None

        self.vectorized_envs = make_vec_envs(
            self.config.env_name,
            self.config.env_configs,
            self.config.random_seed,
            self.config.num_processes,
            self.device,
            self.config.discount_gamma,
            norm_rewards=self.config.norm_rewards,
            norm_observations=self.config.norm_observations,
        )

        self.actor_critic = StatefulActorCritic(
            self.vectorized_envs.observation_space,
            self.vectorized_envs.action_space,
            recurrent_state_size=256,
        ).to_device(self.device)

        self.ppo = PPO(
            actor_critic=self.actor_critic,
            clip_param=self.config.ppo_clip_param,
            opt_epochs=self.config.ppo_opt_epochs,
            num_minibatches=self.config.ppo_num_minibatches,
            value_loss_coef=self.config.ppo_value_loss_coef,
            entropy_coef=self.config.ppo_entropy_coef,
            actor_lr=self.config.actor_lr,
            critic_lr=self.config.critic_lr,
            eps=self.config.optimizer_eps,
            max_grad_norm=self.config.max_grad_norm,
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def wandb_init(self, is_dev: bool) -> None:
        """
        Initalize wandb for the current training run.

        Args:
                is_dev (bool): Whether the current experiment is a trial run.

        Returns:
                None
        """
        wandb.login()
        project_suffix = "-dev" if is_dev else ""

        if self.checkpoint is None or self.checkpoint.wandb_run_id is None:
            wandb.init(
                project=f"autometa{project_suffix}",
                config=self.config.to_dict(),
            )
        else:
            wandb.init(
                project=f"autometa{project_suffix}",
                id=self.checkpoint.wandb_run_id,
            )
            pass

        self.wandb_initialized = True
        pass

    @abstractmethod
    def train(
        self,
        checkpoint_interval: int,
        enable_wandb: bool,
        is_dev: bool = True,
    ) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Args:
            checkpoint_interval (bool): Number of iterations after which to checkpoint.
            enable_wandb (bool): Whether to log to Wandb, `True` by default.
            is_dev (bool): Whether this is a dev run of th experiment.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads relevant info from checkpoint (e.g. current iteration, actor-critic state, optimizer state, etc.)

        Args:
                checkpoint_path (str): Absolute path from which to load the checkpoint.

        Returns:
                BaseTrainingCheckpoint
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self) -> None:
        """
        Save checkpoint.

        Returns:
                None
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """
        Torch device to use for training and optimization.

        Returns:
                torch.device
        """
        if isinstance(self.torch_device, torch.device):
            return self.torch_device

        use_cuda = self.config.use_cuda and torch.cuda.is_available()
        if use_cuda and self.config.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.torch_device = torch.device("cuda:0" if use_cuda else "cpu")

        return self.torch_device
