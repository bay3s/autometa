import csv
import argparse
import os

import numpy as np

import torch

from autometa.training.base_training_config import BaseTrainingConfig
from autometa.sampling.meta_episode_batch import MetaEpisodeBatch

from autometa.utils.env_utils import (
    make_vec_envs,
    register_custom_envs,
    get_vec_normalize,
)

from autometa.networks.stateful.stateful_actor_critic import StatefulActorCritic

register_custom_envs()


if __name__ == "__main__":
    RL_SQUARED = "rl_squared"
    AUTO_DR = "auto_dr"

    EVAL_DIRECTORY = os.path.dirname(__file__)
    MODELS_DIRECTORY = f"{EVAL_DIRECTORY}/models/"
    DATA_DIRECTORY = f"{EVAL_DIRECTORY}/data/"

    SUPPORTED_ALGOS = [RL_SQUARED, AUTO_DR]
    NUM_META_EPISODES = 1_00_00
    NUM_PROCESSES = 25
    RECURRENT_STATE_SIZE = 256
    TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        pass

    parser = argparse.ArgumentParser(
        description="Collect data for evaluation in the point navigation environment."
    )

    parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        default=None,
        help=f"Training algorithm, one of [{', '.join(SUPPORTED_ALGOS)}].",
    )

    parser.add_argument(
        "--grid-size",
        help = "Size of the grid to be used for evaluation.",
        type = float
    )

    parser.add_argument(
        "--random-seed",
        help = "Size of the grid to be used for evaluation.",
        type = int
    )

    args = parser.parse_args()

    # torch seed
    torch.manual_seed(args.random_seed)

    # config
    config_path = f"{MODELS_DIRECTORY}/{args.algo}/config.json"
    configs = BaseTrainingConfig.from_json(config_path)

    # checkpoint
    checkpoint_path = f"{MODELS_DIRECTORY}/{args.algo}/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=TORCH_DEVICE)

    # define tasks
    num_coordinates = int(np.sqrt(NUM_META_EPISODES))
    x_coordinates = np.linspace(-args.grid_size, args.grid_size, num = num_coordinates)
    y_coordinates = np.linspace(-args.grid_size, args.grid_size, num = num_coordinates)

    tasks = list()
    results = dict()

    for i in range(len(x_coordinates)):
        for j in range(len(y_coordinates)):
            tasks.append({
                "x_position": x_coordinates[i],
                "y_position": y_coordinates[j]
            })
            continue

    meta_episode_results = list()

    rl_squared_envs = make_vec_envs(
        env_name=configs.env_id,
        meta_episode_length = configs.meta_episode_length,
        env_kwargs=configs.env_configs,
        seed = args.random_seed,
        num_processes = NUM_PROCESSES,
        device = TORCH_DEVICE,
        gamma = configs.discount_gamma,
        norm_observations = configs.norm_observations,
        norm_rewards = configs.norm_rewards,
    )

    vec_norm = get_vec_normalize(rl_squared_envs)
    if vec_norm is not None:
        vec_norm.obs_rms = checkpoint["observations_rms"]
        vec_norm.ret_rms = checkpoint["rewards_rms"]
        pass

    # policy
    actor_critic = StatefulActorCritic(
        rl_squared_envs.observation_space,
        rl_squared_envs.action_space,
        recurrent_state_size=RECURRENT_STATE_SIZE,
    ).to_device(TORCH_DEVICE)

    actor_critic.actor.load_state_dict(checkpoint["actor_state_dict"])
    actor_critic.critic.load_state_dict(checkpoint["critic_state_dict"])

    # render
    recurrent_states_actor = torch.zeros(1, actor_critic.recurrent_state_size)
    recurrent_states_critic = torch.zeros(1, actor_critic.recurrent_state_size)
    recurrent_masks = torch.zeros(1, 1)

    obs = rl_squared_envs.reset()

    with torch.no_grad():
        for iters in range(NUM_META_EPISODES // NUM_PROCESSES):
            print("Current seed:", args.random_seed, ", number of meta episodes:", iters * NUM_PROCESSES, " of ",
                  NUM_META_EPISODES)
            meta_episodes = MetaEpisodeBatch(
                configs.meta_episode_length,
                NUM_PROCESSES,
                rl_squared_envs.observation_space,
                rl_squared_envs.action_space,
                RECURRENT_STATE_SIZE,
            )

            sampled_tasks = tasks[iters * NUM_PROCESSES: iters * NUM_PROCESSES + NUM_PROCESSES]
            rl_squared_envs.sample_tasks_async(sampled_tasks)
            initial_observations = rl_squared_envs.reset()
            meta_episodes.obs[0].copy_(initial_observations)

            for step in range(configs.meta_episode_length):
                (
                    value_preds,
                    actions,
                    action_log_probs,
                    recurrent_states_actor,
                    recurrent_states_critic,
                ) = actor_critic.act(
                    meta_episodes.obs[step].to(TORCH_DEVICE),
                    meta_episodes.recurrent_states_actor[step].to(TORCH_DEVICE),
                    meta_episodes.recurrent_states_critic[step].to(TORCH_DEVICE),
                    recurrent_state_masks = None,
                    deterministic = True
                )

                obs, rewards, dones, infos = rl_squared_envs.step(actions)

                # rewards
                for info in infos:
                    if "meta_episode" in info.keys():
                        meta_episode_results.append({
                            "x_position": info["meta_episode"]["sampled_task"][0],
                            "y_position": info["meta_episode"]["sampled_task"][1],
                            "r": info["meta_episode"]["r"],
                            "random_seed": args.random_seed
                        })
                        pass

                # dones
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
                continue

    # save
    keys = meta_episode_results[0].keys()

    # results
    results_directory = f"{DATA_DIRECTORY}/{args.algo}/{int(args.grid_size)}x{int(args.grid_size)}/"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    with open(f"{results_directory}/seed-{args.random_seed}.csv", "w", newline = "") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(meta_episode_results)
        pass
