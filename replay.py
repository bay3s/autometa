import argparse
import os
import torch

from autometa.training.base_training_config import BaseTrainingConfig

from autometa.utils.env_utils import (
    make_vec_envs,
    register_custom_envs,
    get_vec_normalize,
)

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

    parser.add_argument("--seed", help="Random seed to be used.", type=int)

    parser.add_argument(
        "--run", help="`wandb` run id used for model training.", type=str
    )

    parser.add_argument("--length", help="`episode_length` for the replay.", type=int)

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
    config_path = f"{MODEL_DIRECTORY}/{env_folder}/{args.run}/config.json"
    configs = BaseTrainingConfig.from_json(config_path)

    # checkpoint
    checkpoint_path = f"{MODEL_DIRECTORY}/{env_folder}/{args.run}/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    configs.env_configs["episode_length"] = args.length

    vectorized_envs = make_vec_envs(
        env_name=configs.env_name,
        meta_episode_length = configs.meta_episode_length,
        env_kwargs=configs.env_configs,
        seed = args.seed,
        num_processes = 1,
        device = torch.device("cpu"),
        gamma = configs.discount_gamma,
        norm_observations = configs.norm_observations,
        norm_rewards = configs.norm_rewards,
    )

    vec_norm = get_vec_normalize(vectorized_envs)
    if vec_norm is not None:
        # @todo update naming conventions for checkpointing.
        vec_norm.obs_rms = checkpoint["observations_rms"]
        vec_norm.ret_rms = checkpoint["reward_rms"]
        pass

    # policy
    actor_critic = StatefulActorCritic(
        vectorized_envs.observation_space,
        vectorized_envs.action_space,
        recurrent_state_size=256,
    ).to_device(torch.device("cpu"))

    actor_critic.actor.load_state_dict(checkpoint["actor"])
    actor_critic.critic.load_state_dict(checkpoint["critic"])

    # render
    vectorized_envs.reset()

    recurrent_states_actor = torch.zeros(1, actor_critic.recurrent_state_size)
    recurrent_states_critic = torch.zeros(1, actor_critic.recurrent_state_size)
    recurrent_masks = torch.zeros(1, 1)

    obs = vectorized_envs.reset()
    starting_obs = obs
    vectorized_envs.env_method("render", indices=[0], mode="human")

    while True:
        with torch.no_grad():
            (
                value,
                action,
                _,
                recurrent_states_actor,
                recurrent_states_critic,
            ) = actor_critic.act(
                observations=obs,
                recurrent_states_actor=recurrent_states_actor,
                recurrent_states_critic=recurrent_states_critic,
                recurrent_state_masks=recurrent_masks,
                deterministic=args.deterministic,
            )
            pass

        prev_obs = obs
        obs, reward, done, _ = vectorized_envs.step(action)
        vectorized_envs.env_method("render", indices=[0], mode="human")

        if done[0]:
            print("done")
            continue
