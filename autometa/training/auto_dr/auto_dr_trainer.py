import os

import torch
import wandb

import autometa.utils.logging_utils as logging_utils
from autometa.training.auto_dr.auto_dr_config import AutoDRConfig
from autometa.learners.ppo import PPO

from autometa.utils.env_utils import make_vec_envs, get_vec_normalize
from autometa.utils.training_utils import (
    sample_auto_dr,
    timestamp,
)

from autometa.training.training_checkpoint import TrainingCheckpoint
from autometa.sampling.meta_batch_sampler import MetaBatchSampler
from autometa.networks.stateful.stateful_actor_critic import StatefulActorCritic
from autometa.randomization.randomizer import Randomizer


class AutoDRTrainer:
    def __init__(self, config: AutoDRConfig = None, checkpoint_path: str = None):
        """
        Initialize an instance of a trainer for PPO.

        Args:
            config (AutoDRConfig): Params to be used for the trainer.
            checkpoint_path (str): TrainingCheckpoint path from where to restart the experiment.
        """
        self.config = config

        # checkpoint
        if checkpoint_path is not None:
            self.restart_checkpoint = TrainingCheckpoint.load(
                checkpoint_path, self.device
            )
        else:
            self.restart_checkpoint = None
            pass

        # private
        self._device = None
        self._log_dir = None

        # general
        self.ppo = None
        self.vectorized_envs = None
        self.actor_critic = None

        # randomization
        self.randomizer = None
        pass

    def train(
        self,
        checkpoint_interval: int = 1,
        enable_wandb: bool = True,
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
        # log
        self.config.save()

        if enable_wandb:
            wandb.login()
            project_suffix = "-dev" if is_dev else ""

            if self.restart_checkpoint is None or self.config.wandb_run_id is None:
                wandb_run_id = wandb.util.generate_id()
                self.config.wandb_run_id = wandb_run_id
                wandb.init(
                    project=f"autometa{project_suffix}",
                    config=self.config.dict,
                    id=self.config.wandb_run_id,
                )
            else:
                wandb.init(
                    project=f"autometa{project_suffix}", id=self.config.wandb_run_id
                )
                pass

        # seed
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)

        # clean
        logging_utils.cleanup_log_dir(self.log_dir)
        torch.set_num_threads(1)

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

        self.randomizer = Randomizer(
            parallel_envs=self.vectorized_envs,
            evaluation_probability=self.config.adr_evaluation_probability,
            buffer_size=self.config.adr_performance_buffer_size,
            performance_threshold_lower=self.config.adr_performance_threshold_lower,
            performance_threshold_upper=self.config.adr_performance_threshold_upper,
            delta=self.config.adr_delta,
        )

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

        current_iteration = 0

        # load
        if self.restart_checkpoint:
            current_iteration = self.restart_checkpoint.current_iteration

            # policy / ppo
            self.actor_critic.actor.load_state_dict(self.restart_checkpoint.actor_state_dict)
            self.actor_critic.critic.load_state_dict(
                self.restart_checkpoint.critic_state_dict
            )
            self.ppo.optimizer.load_state_dict(self.restart_checkpoint.optimizer_state_dict)

            # rms
            vec_normalized = get_vec_normalize(self.vectorized_envs)
            vec_normalized.obs_rms = self.restart_checkpoint.observations_rms
            vec_normalized.ret_rms = self.restart_checkpoint.rewards_rms
            pass

        for j in range(current_iteration, self.config.policy_iterations):
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
                self.checkpoint(current_iteration)
                pass

            if enable_wandb:
                wandb.log(wandb_logs)

        if enable_wandb:
            wandb.finish()

    def checkpoint(self, current_iteration: int) -> None:
        """
        Save checkpoint.

        Args:
            current_iteration (int): Current training iteration.

        Returns:
            None
        """
        vec_normalized = get_vec_normalize(self.vectorized_envs)
        checkpoint_name = str(timestamp()) if self.config.checkpoint_all else ""

        checkpoint = TrainingCheckpoint(
            current_iteration = current_iteration,
            actor_state_dict = self.actor_critic.actor.state_dict(),
            critic_state_dict = self.actor_critic.critic.state_dict(),
            optimizer_state_dict = self.ppo.optimizer.state_dict(),
            observations_rms = (
                vec_normalized.obs_rms if vec_normalized is not None else None
            ),
            rewards_rms = (
                vec_normalized.ret_rms if vec_normalized is not None else None
            ),
        )
        checkpoint.save(self.config.checkpoint_dir, checkpoint_name)
        pass

    @property
    def log_dir(self) -> str:
        """
        Returns the path for training logs.

        Returns:
            str
        """
        if not self._log_dir:
            self._log_dir = os.path.expanduser(self.config.log_dir)

        return self._log_dir

    @property
    def device(self) -> torch.device:
        """
        Torch device to use for training and optimization.

        Returns:
          torch.device
        """
        if isinstance(self._device, torch.device):
            return self._device

        use_cuda = self.config.use_cuda and torch.cuda.is_available()
        if use_cuda and self.config.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        return self._device
