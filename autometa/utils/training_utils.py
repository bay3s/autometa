from typing import List, Tuple
from datetime import datetime

import torch

from autometa.envs.pytorch_vec_env_wrapper import PyTorchVecEnvWrapper
from autometa.networks.stateful.stateful_actor_critic import StatefulActorCritic
from autometa.sampling.meta_episode_batch import MetaEpisodeBatch
from autometa.randomization.randomizer import Randomizer


@torch.no_grad()
def sample_auto_dr(
    randomizer: Randomizer,
    actor_critic: StatefulActorCritic,
    meta_episode_length: int,
    num_meta_episodes: int,
    use_gae: bool,
    gae_lambda: float,
    discount_gamma: float,
    device: torch.device,
) -> Tuple[List[MetaEpisodeBatch], List]:
    """
    Sample meta-episodes in parallel.

    Returns a list of meta-episodes and the mean reward per step.

    Args:
        randomizer (Randomizer): Domain randomizer used for the experiment.
        actor_critic (StatefulActorCritic): Actor-critic to be used for sampling.
        meta_episode_length (int): Meta-episode length, each "meta-episode" has multiple episodes.
        num_meta_episodes (int): Number of meta-episodes to sample.
        use_gae (bool): Whether to use GAE to compute advantages.
        gae_lambda (float): GAE lambda parameter.
        discount_gamma (float): Discount rate.
        device (torch.device): Device on which to transfer the tensors.

    Returns:
        Tuple[List[MetaEpisodeBatch], float]
    """
    observation_space = randomizer.observation_space
    action_space = randomizer.action_space

    recurrent_state_size = actor_critic.recurrent_state_size
    num_parallel_envs = randomizer.num_envs

    meta_episode_batch = list()
    meta_episode_rewards = list()

    for _ in range(num_meta_episodes // num_parallel_envs):
        meta_episodes = MetaEpisodeBatch(
            meta_episode_length,
            num_parallel_envs,
            observation_space,
            action_space,
            recurrent_state_size,
        )

        randomizer.re_evaluate()
        randomizer.randomize_all()

        initial_observations = randomizer.parallel_envs.reset()
        meta_episodes.obs[0].copy_(initial_observations)

        for step in range(meta_episode_length):
            (
                value_preds,
                actions,
                action_log_probs,
                recurrent_states_actor,
                recurrent_states_critic,
            ) = actor_critic.act(
                meta_episodes.obs[step].to(device),
                meta_episodes.recurrent_states_actor[step].to(device),
                meta_episodes.recurrent_states_critic[step].to(device),
            )

            obs, rewards, dones, infos = randomizer.step(actions)

            # rewards
            for info in infos:
                if "meta_episode" in info.keys():
                    meta_episode_rewards.append(info["meta_episode"]["r"])

            done_masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in dones]
            )

            # insert
            meta_episodes.insert(
                obs,
                recurrent_states_actor,
                recurrent_states_critic,
                actions,
                action_log_probs,
                value_preds,
                rewards,
                done_masks,
            )
            pass

        next_value_pred, _ = actor_critic.get_value(
            meta_episodes.obs[-1].to(device),
            meta_episodes.recurrent_states_critic[-1].to(device),
        )

        next_value_pred.detach()

        meta_episodes.compute_returns(
            next_value_pred, use_gae, discount_gamma, gae_lambda
        )

        meta_episode_batch.append(meta_episodes)
        pass

    return meta_episode_batch, meta_episode_rewards


@torch.no_grad()
def sample_rl_squared(
    actor_critic: StatefulActorCritic,
    parallel_envs: PyTorchVecEnvWrapper,
    meta_episode_length: int,
    num_meta_episodes: int,
    use_gae: bool,
    gae_lambda: float,
    discount_gamma: float,
    device: torch.device,
) -> Tuple[List[MetaEpisodeBatch], List]:
    """
    Sample meta-episodes in parallel.

    Returns a list of meta-episodes and the mean reward per step.

    Args:
        actor_critic (StatefulActorCritic): Actor-critic to be used for sampling.
        parallel_envs (PyTorchVecEnvWrapper): Parallel environments for sampling episodes.
        meta_episode_length (int): Meta-episode length, each "meta-episode" has multiple episodes.
        num_meta_episodes (int): Number of meta-episodes to sample.
        use_gae (bool): Whether to use GAE to compute advantages.
        gae_lambda (float): GAE lambda parameter.
        discount_gamma (float): Discount rate.
        device (torch.device): Device on which to transfer the tensors.

    Returns:
        Tuple[List[MetaEpisodeBatch], float]
    """
    observation_space = parallel_envs.observation_space
    action_space = parallel_envs.action_space

    recurrent_state_size = actor_critic.recurrent_state_size
    num_parallel_envs = parallel_envs.num_envs

    meta_episode_batch = list()
    meta_episode_rewards = list()

    for _ in range(num_meta_episodes // num_parallel_envs):
        meta_episodes = MetaEpisodeBatch(
            meta_episode_length,
            num_parallel_envs,
            observation_space,
            action_space,
            recurrent_state_size,
        )

        parallel_envs.sample_tasks_async()
        initial_observations = parallel_envs.reset()
        meta_episodes.obs[0].copy_(initial_observations)

        for step in range(meta_episode_length):
            (
                value_preds,
                actions,
                action_log_probs,
                recurrent_states_actor,
                recurrent_states_critic,
            ) = actor_critic.act(
                meta_episodes.obs[step].to(device),
                meta_episodes.recurrent_states_actor[step].to(device),
                meta_episodes.recurrent_states_critic[step].to(device),
            )

            obs, rewards, dones, infos = parallel_envs.step(actions)

            # rewards
            for info in infos:
                if "meta_episode" in info.keys():
                    meta_episode_rewards.append(info["meta_episode"]["r"])

            done_masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in dones]
            )

            # insert
            meta_episodes.insert(
                obs,
                recurrent_states_actor,
                recurrent_states_critic,
                actions,
                action_log_probs,
                value_preds,
                rewards,
                done_masks,
            )
            pass

        next_value_pred, _ = actor_critic.get_value(
            meta_episodes.obs[-1].to(device),
            meta_episodes.recurrent_states_critic[-1].to(device),
        )

        next_value_pred.detach()

        meta_episodes.compute_returns(
            next_value_pred, use_gae, discount_gamma, gae_lambda
        )

        meta_episode_batch.append(meta_episodes)
        pass

    return meta_episode_batch, meta_episode_rewards


def evaluate(
    actor_critic: StatefulActorCritic,
    parallel_envs: PyTorchVecEnvWrapper,
    meta_episode_length: int,
    num_meta_episodes: int,
    device: torch.device,
) -> Tuple[List[MetaEpisodeBatch], List]:
    """
    Sample meta-episodes in parallel.

    Returns a list of meta-episodes and the mean reward per step.

    Args:
        actor_critic (StatefulActorCritic): Actor-critic to be used for sampling.
        parallel_envs (PyTorchVecEnvWrapper): Parallel environments for sampling episodes.
        meta_episode_length (int): Meta-episode length, each "meta-episode" has multiple episodes.
        num_meta_episodes (int): Number of meta-episodes to sample.
        device (torch.device): Device on which to transfer the tensors.

    Returns:
        Tuple[List[MetaEpisodeBatch], float]
    """
    observation_space = parallel_envs.observation_space
    action_space = parallel_envs.action_space

    recurrent_state_size = actor_critic.recurrent_state_size
    num_parallel_envs = parallel_envs.num_envs

    meta_episode_batch = list()
    meta_episode_rewards = list()

    actor_critic.actor.eval()
    actor_critic.critic.eval()

    for _ in range(num_meta_episodes // num_parallel_envs):
        meta_episodes = MetaEpisodeBatch(
            meta_episode_length,
            num_parallel_envs,
            observation_space,
            action_space,
            recurrent_state_size,
        )

        parallel_envs.sample_tasks_async()
        initial_observations = parallel_envs.reset()
        meta_episodes.obs[0].copy_(initial_observations)

        for step in range(meta_episode_length):
            (
                value_preds,
                actions,
                action_log_probs,
                recurrent_states_actor,
                recurrent_states_critic,
            ) = actor_critic.act(
                meta_episodes.obs[step].to(device),
                meta_episodes.recurrent_states_actor[step].to(device),
                meta_episodes.recurrent_states_critic[step].to(device),
            )

            obs, rewards, dones, infos = parallel_envs.step(actions)

            # rewards
            for info in infos:
                if "meta_episode" in info.keys():
                    meta_episode_rewards.append(info["meta_episode"]["r"])

            done_masks = torch.FloatTensor(
                [[0.0] if _done else [1.0] for _done in dones]
            )

            # insert
            meta_episodes.insert(
                obs,
                recurrent_states_actor,
                recurrent_states_critic,
                actions,
                action_log_probs,
                value_preds,
                rewards,
                done_masks,
            )
            pass

        meta_episode_batch.append(meta_episodes)
        pass

    actor_critic.actor.train()
    actor_critic.critic.train()

    return meta_episode_batch, meta_episode_rewards


def timestamp() -> int:
    """
    Return the current timestamp in integer format.

    Returns:
        int
    """
    return int(datetime.timestamp(datetime.now()))
