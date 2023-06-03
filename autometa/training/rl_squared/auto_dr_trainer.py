import torch
import wandb

from autometa.training.base_trainer import BaseTrainer
from autometa.training.auto_dr.auto_dr_config import AutoDRConfig

from autometa.utils.env_utils import get_vec_normalize
from autometa.training.auto_dr.auto_dr_checkpoint import AutoDRCheckpoint
from autometa.utils.training_utils import (
    sample_auto_dr,
    timestamp,
)

from autometa.sampling.meta_batch_sampler import MetaBatchSampler
from autometa.randomization.randomizer import Randomizer


class AutoDRTrainer(BaseTrainer):
    def __init__(self, config: AutoDRConfig, checkpoint_path: str = None):
        """
        Initialize an instance of a trainer for PPO.

        Args:
            config (AutoDRConfig): Params to be used for the trainer.
            checkpoint_path (str): Checkpoint path from which to restart.
        """
        super().__init__(config, checkpoint_path)

        self.randomizer = Randomizer(
            parallel_envs=self.vectorized_envs,
            evaluation_probability=self.config.adr_evaluation_probability,
            buffer_size=self.config.adr_performance_buffer_size,
            performance_threshold_lower=self.config.adr_performance_threshold_lower,
            performance_threshold_upper=self.config.adr_performance_threshold_upper,
            delta=self.config.adr_delta,
        )
        pass

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads relevant info from checkpoint (eg. current iteration, actor-critic state, optimizer state, etc.)

        Args:
            checkpoint_path (str): Absolute path from which to load the checkpoint.

        Returns:
            None
        """
        self.checkpoint = AutoDRCheckpoint.load(checkpoint_path, self.device)
        self.current_iteration = self.checkpoint.current_iteration

        # policy / ppo
        self.actor_critic.actor.load_state_dict(self.checkpoint.actor_state_dict)
        self.actor_critic.critic.load_state_dict(self.checkpoint.critic_state_dict)
        self.ppo.optimizer.load_state_dict(self.checkpoint.optimizer_state_dict)

        # rms
        vec_normalized = get_vec_normalize(self.vectorized_envs)
        vec_normalized.obs_rms = self.checkpoint.observations_rms
        vec_normalized.ret_rms = self.checkpoint.rewards_rms

        # randomizer
        self.randomizer.randomized_parameters = self.checkpoint.randomized_parameters
        self.randomizer.buffer = self.checkpoint.randomization_buffer
        pass

    def train(
        self,
        checkpoint_interval: int,
        checkpoint_all: bool,
        enable_wandb: bool,
        is_dev: bool = True,
    ) -> None:
        """
        Train an agent based on the configs specified by the training parameters.

        Args:
            checkpoint_interval (int): Number of iterations after which to checkpoint.
            checkpoint_all (bool): Whether to archive all checkpoints.
            enable_wandb (bool): Whether to log to Wandb, `True` by default.
            is_dev (bool): Whether this is a dev run of th experiment.

        Returns:
            None
        """
        # enable wandb
        if enable_wandb:
            self.wandb_init(is_dev)

        # save
        self.save_config()

        # seed
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        torch.set_num_threads(1)

        for j in range(self.current_iteration, self.config.policy_iterations):
            # anneal
            if self.config.use_linear_lr_decay:
                self.ppo.anneal_learning_rates(j, self.config.policy_iterations)
                pass

            # sample
            meta_episode_batches, meta_train_reward_per_step = sample_auto_dr(
                self.randomizer,
                self.actor_critic,
                self.config.meta_episode_length,
                self.config.meta_episodes_per_epoch,
                self.config.use_gae,
                self.config.gae_lambda,
                self.config.discount_gamma,
                self.device,
            )

            minibatch_sampler = MetaBatchSampler(meta_episode_batches, self.device)
            ppo_update = self.ppo.update(minibatch_sampler)

            wandb_logs = {
                "meta_train/mean_policy_loss": ppo_update.policy_loss,
                "meta_train/mean_value_loss": ppo_update.value_loss,
                "meta_train/mean_entropy": ppo_update.entropy,
                "meta_train/approx_kl": ppo_update.approx_kl,
                "meta_train/clip_fraction": ppo_update.clip_fraction,
                "meta_train/explained_variance": ppo_update.explained_variance,
                "meta_train/mean_meta_episode_reward": meta_train_reward_per_step
                * self.config.meta_episode_length,
            }

            # add
            wandb_logs.update(self.randomizer.info)

            # checkpoint
            is_last_iteration = j == (self.config.policy_iterations - 1)

            if j % checkpoint_interval == 0 or is_last_iteration:
                checkpoint_name = str(timestamp()) if checkpoint_all else ""
                self.save_checkpoint(checkpoint_name)
                pass

            if enable_wandb:
                wandb.log(wandb_logs)

            self.current_iteration = j + 1
            continue

        # end
        if enable_wandb:
            wandb.finish()
        pass

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save checkpoint.

         Args:
            checkpoint_name (str): Checkpoint name to be used while saving.

        Returns:
            None
        """
        vec_normalized = get_vec_normalize(self.vectorized_envs)

        checkpoint = AutoDRCheckpoint(
            wandb_run_id = (
                wandb.run.id if (self.wandb_initialized and wandb and wandb.run) else None
            ),
            current_iteration = self.current_iteration,
            actor_state_dict = self.actor_critic.actor.state_dict(),
            critic_state_dict = self.actor_critic.critic.state_dict(),
            optimizer_state_dict = self.ppo.optimizer.state_dict(),
            observations_rms = (
                vec_normalized.obs_rms if vec_normalized is not None else None
            ),
            rewards_rms = (
                vec_normalized.ret_rms if vec_normalized is not None else None
            ),
            randomized_parameters = self.randomizer.randomized_parameters,
            randomization_buffer = self.randomizer.buffer,
        )

        checkpoint.save(self.checkpoint_directory, checkpoint_name)
        pass