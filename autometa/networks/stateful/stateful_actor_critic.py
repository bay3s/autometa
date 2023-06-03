from typing import Tuple, Union, List

import torch
import torch.nn as nn
import gym

from autometa.networks.base_actor_critic import BaseActorCritic
from autometa.utils.torch_utils import init_orthogonal
from autometa.networks.modules.distributions import Categorical, DiagonalGaussian
from autometa.networks.modules.memory.gru import GRU
from autometa.utils.torch_utils import init_mlp


class StatefulActorCritic(nn.Module, BaseActorCritic):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        recurrent_state_size: int,
        hidden_sizes: List = [256, 256],
    ):
        """
        Actor-Critic for a discrete action space.

        Args:
          observation_space (gym.Space): Observation space in which the agent operates.
          action_space (gym.Space): Action space in which the agent operates.
          recurrent_state_size (int): Recurrent state size.
        """
        nn.Module.__init__(self)
        BaseActorCritic.__init__(self, observation_space, action_space)

        # base
        self.gru = GRU(observation_space.shape[0], recurrent_state_size)

        # actor
        self.actor = init_mlp(recurrent_state_size, hidden_sizes=hidden_sizes)
        self.actor_dist = self._init_dist(hidden_sizes[-1], action_space=action_space)

        # critic
        self.critic = init_mlp(recurrent_state_size, hidden_sizes=hidden_sizes)
        self.critic_linear = init_orthogonal(nn.Linear(hidden_sizes[-1], 1))

        # private
        self._recurrent_state_size = recurrent_state_size
        self._device = None
        pass

    def base(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Conduct the forward pass for the base modules for the actor critic.

        Args:
            observations (torch.Tensor): State in which to take an action.
            recurrent_states (torch.Tensor): Recurrent states for the actor-critic.
            recurrent_state_masks (torch.Tensor): Recurrent states masks for the actor-critic.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        x, recurrent_states = self.gru(
            observations, recurrent_states, recurrent_state_masks, self._device
        )

        actor_features = self.actor(x)
        critic_features = self.critic(x)
        value_estimates = self.critic_linear(critic_features)

        return value_estimates, actor_features, recurrent_states

    def act(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return the action to take, log probability of said action, and the current state value
        computed by the critic.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states for the actor-critic.
          recurrent_state_masks (torch.Tensor): Recurrent states masks for the actor-critic.
          deterministic (bool): Whether to choose actions deterministically.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        value_estimates, actor_features, recurrent_states = self.base(
            observations, recurrent_states, recurrent_state_masks
        )

        action_distribution = self.actor_dist(actor_features)

        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )

        return (
            value_estimates,
            actions,
            action_distribution.log_probs(actions),
            recurrent_states,
        )

    def get_value(
        self,
        observations: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Given a state returns its corresponding value.

        Args:
          observations (torch.Tensor): State in which to take an action.
          recurrent_states (torch.Tensor): Recurrent states that are being used in memory-based policies.
          recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.

        Returns:
          torch.Tensor
        """
        value_estimates, _, _ = self.base(observations, recurrent_states, recurrent_state_masks)

        return value_estimates

    def evaluate_actions(
        self,
        inputs: torch.Tensor,
        actions: torch.Tensor,
        recurrent_states: torch.Tensor,
        recurrent_state_masks: torch.Tensor = None,
    ) -> Tuple:
        """
        Evaluate actions given observations, states, actions, and recurrent state masks.

        Args:
            inputs (torch.Tensor): Inputs to the actor and the critic.
            actions (torch.Tensor): Actions taken at each timestep.
            recurrent_states (torch.Tensor): Recurrent states for the actor-critic.
            recurrent_state_masks (torch.Tensor): Masks to be applied to the recurrent states.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        value_estimates, actor_features, recurrent_states = self.base(inputs, recurrent_states, recurrent_state_masks)
        action_distribution = self.actor_dist(actor_features)

        action_log_probs = action_distribution.log_probs(actions)
        dist_entropy = action_distribution.entropy().mean()

        return value_estimates, action_log_probs, dist_entropy, recurrent_states

    @property
    def recurrent_state_size(self) -> int:
        """
        Returns the size of the encoded state (eg. hidden state in a recurrent agent).

        Returns:
          int
        """
        return self._recurrent_state_size

    def _init_dist(
        self, last_hidden_size: int, action_space: gym.Space
    ) -> Union[Categorical, DiagonalGaussian]:
        """
        Initialize the action distribution.

        Args:
            last_hidden_size (int): Size of the last hidden layer in the MLP.
            action_space (gym.Space): Action space for the actor.

        Returns:
            Union[Categorical, DiagonalGaussian]
        """
        if action_space.__class__.__name__ == "Discrete":
            return Categorical(last_hidden_size, action_space.n)
        elif action_space.__class__.__name__ == "Box":
            return DiagonalGaussian(last_hidden_size, action_space.shape[0])
        else:
            raise NotImplementedError

    def to_device(self, device: torch.device) -> "BaseActorCritic":
        """
        Performse device conversion on the actor and critic.

        Returns:
          BaseCritic
        """
        self._device = device

        return self.to(device)

    def forward(self) -> None:
        """
        Forward pass for the network, in this case not implemented.

        Returns:
          None
        """
        raise NotImplementedError
