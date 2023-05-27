import os

import torch
import wandb

import autometa.utils.logging_utils as logging_utils
from autometa.training.auto_dr.auto_dr_config import AutoDRConfig
from autometa.learners.ppo import PPO

from autometa.utils.env_utils import make_vec_envs, get_vec_normalize
from autometa.utils.training_utils import (
    sample_auto_dr,
    save_checkpoint,
    timestamp,
)

from autometa.sampling.meta_batch_sampler import MetaBatchSampler
from autometa.networks.stateful.stateful_actor_critic import StatefulActorCritic
from autometa.randomization.randomizer import Randomizer


class AutoDRTrainer:
    def __init__(self, config: AutoDRConfig, checkpoint_path: str = None):
        """
        Initialize an instance of a trainer for PPO.

        Args:
            config (AutoDRConfig): Params to be used for the trainer.
            checkpoint_path (str): Checkpoint path from where to restart the experiment.
        """
        self.config = config

        # private
        self._device = None
        self._log_dir = None

        # general
        self.ppo = None
        self.vectorized_envs = None
        self.actor_critic = None

        # domain randomization
        self.randomizer = None
        self.sampled_boundaries = None

        # checkpoint
        self._checkpoint_path = checkpoint_path
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
        self.save_params()

        if enable_wandb:
            wandb.login()
            project_suffix = "-dev" if is_dev else ""
            wandb.init(project=f"autometa{project_suffix}", config=self.config.dict)
            self.config.run_id = wandb.run.id
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
        if self._checkpoint_path:
            checkpoint = torch.load(self._checkpoint_path, map_location = self.device)
            current_iteration = checkpoint["iteration"]

            # policy / ppo
            self.actor_critic.actor.load_state_dict(checkpoint["actor"])
            self.actor_critic.critic.load_state_dict(checkpoint["critic"])
            self.ppo.optimizer.load_state_dict(checkpoint["optimizer"])

            # rms
            vec_normalized = get_vec_normalize(self.vectorized_envs)
            vec_normalized.obs_rms = checkpoint["observation_rms"]
            vec_normalized.ret_rms = checkpoint["reward_rms"]
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
            checkpoint_ver = str(timestamp()) if self.config.checkpoint_all else ""
            is_last_iteration = j == (self.config.policy_iterations - 1)

            if j % checkpoint_interval == 0 or is_last_iteration:
                vec_normalized = get_vec_normalize(self.vectorized_envs)

                save_checkpoint(
                    iteration=j,
                    checkpoint_dir=self.config.checkpoint_dir,
                    checkpoint_name=checkpoint_ver,
                    actor=self.actor_critic.actor,
                    critic=self.actor_critic.critic,
                    optimizer=self.ppo.optimizer,
                    observations_rms=(
                        vec_normalized.obs_rms if vec_normalized is not None else None
                    ),
                    rewards_rms=(
                        vec_normalized.ret_rms if vec_normalized is not None else None
                    ),
                )
                pass

            if enable_wandb:
                wandb.log(wandb_logs)

        if enable_wandb:
            wandb.finish()

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

    def save_params(self) -> None:
        """
        Save experiment_config to the logging directory.

        Returns:
          None
        """
        self.config.save()
        pass

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
