import argparse
import os
import torch

from autometa.training.base_config import BaseConfig

from autometa.utils.env_utils import make_vec_envs, register_custom_envs
from autometa.networks.stateful.stateful_actor_critic import StatefulActorCritic

register_custom_envs()


if __name__ == "__main__":

    MODEL_DIRECTORY = f"{os.path.dirname(__file__)}/trained_models/"

    RL_SQUARED = "rl_squared"
    AUTO_DR = "auto_dr"

    SUPPORTED_ALGOS = [RL_SQUARED, AUTO_DR]

    parser = argparse.ArgumentParser(
        description="Generate & record replays of the environment."
    )

    parser.add_argument(
        "--env-name", help="Environment for which to run the replay.", type=str
    )

    parser.add_argument(
        "--seed", help="Random seed to be used.", type=int
    )

    parser.add_argument(
        "--run-id", help="`wandb` run id for the model training.", type=str
    )

    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to act deterministically in the environment.",
    )

    args = parser.parse_args()

    env_folder = args.env_name
    for pos, ch in enumerate(env_folder):
        if ch.isupper() and pos > 0:
            env_folder = env_folder.replace(ch, "-%s" % ch.lower())
        elif ch.isupper():
            env_folder = env_folder.replace(ch, ch.lower())

    # config
    config_path = f"{MODEL_DIRECTORY}/{env_folder}/{args.run_id}/config.json"
    configs = BaseConfig.from_json(config_path)

    # checkpoint
    checkpoint_path = f"{MODEL_DIRECTORY}/{env_folder}/{args.model_version}/model.pt"
    checkpoint = torch.load(checkpoint_path)

    vectorized_envs = make_vec_envs(
        configs.env_name,
        configs.env_configs,
        args.seed,
        1,
        torch.device("cpu"),
        configs.discount_gamma,
        # @todo update ADR configs to account for normalization
        configs.norm_observations,
        configs.norm_rewards,
    )

    # policy
    actor_critic = StatefulActorCritic(
        vectorized_envs.observation_space,
        vectorized_envs.action_space,
        recurrent_state_size = 256,
    ).to_device(torch.device("cpu"))

    actor_critic.actor.load_state_dict(checkpoint["actor"])
    actor_critic.critic.load_state_dict(checkpoint["critic"])

    # render
    vectorized_envs.reset()

    recurrent_states_actor = torch.zeros(1, actor_critic.recurrent_state_size)
    recurrent_states_critic = torch.zeros(1, actor_critic.recurrent_state_size)
    recurrent_masks = torch.zeros(1, 1)

    obs = vectorized_envs.reset()
    vectorized_envs.env_method('render', indices=[0], mode = "human")

    while True:
        with torch.no_grad():
            value, action, _, recurrent_states_actor, recurrent_states_critic = actor_critic.act(
                observations=obs,
                recurrent_states_actor=recurrent_states_actor,
                recurrent_states_critic=recurrent_states_critic,
                recurrent_state_masks=recurrent_masks,
                deterministic=args.deterministic,
            )

        obs, reward, done, _ = vectorized_envs.step(action)
        recurrent_states_actor.fill_(0.0 if done else 1.0)
        recurrent_states_critic.fill_(0.0 if done else 1.0)
        vectorized_envs.env_method('render', indices = [0], mode = "human")
        continue

