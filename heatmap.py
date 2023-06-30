import csv
import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    MODEL_DIRECTORY = f"{os.path.dirname(__file__)}/results/"

    RL_SQUARED = "rl_squared"
    AUTO_DR = "auto_dr"

    SUPPORTED_ALGOS = [RL_SQUARED, AUTO_DR]
    NUM_META_EPISODES = 10000
    NUM_PROCESSES = 10
    RECURRENT_STATE_SIZE = 256
    TORCH_DEVICE = torch.device("cpu")

    parser = argparse.ArgumentParser(
        description="Generate & record replays of the environment."
    )

    parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        default=None,
        help=f"Training algorithm, one of [{', '.join(SUPPORTED_ALGOS)}].",
    )

    parser.add_argument(
        "--env", help="Environment for which to run the replay.", type=str
    )

    parser.add_argument(
        "--run", help="`wandb` run id used for model training.", type=str
    )

    parser.add_argument(
        "--num-tasks", help = "Number of tasks to evaluate this over.", type = str
    )

    parser.add_argument(
        "--seed", help = "Random seed to be used for the heatmap.", type = int
    )

    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to act deterministically in the environment.",
    )

    args = parser.parse_args()

    env_folder = args.env
    for pos, ch in enumerate(env_folder):
        if ch.isupper() and pos > 0:
            env_folder = env_folder.replace(ch, "-%s" % ch.lower())
        elif ch.isupper():
            env_folder = env_folder.replace(ch, ch.lower())

    # config
    config_path = f"{MODEL_DIRECTORY}/{args.algo}/{env_folder}/run-{args.run}/config.json"
    configs = BaseTrainingConfig.from_json(config_path)

    # checkpoint
    checkpoint_path = f"{MODEL_DIRECTORY}/{args.algo}/{env_folder}/run-{args.run}/checkpoints/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=TORCH_DEVICE)

    rl_squared_envs = make_vec_envs(
        env_name=configs.env_id,
        meta_episode_length = configs.meta_episode_length,
        env_kwargs=configs.env_configs,
        seed = args.seed,
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

    obs = rl_squared_envs.reset(args.seed)
    meta_episode_results = list()

    # define
    x_coordinates = np.linspace(-3.0, 3.0, num=int(np.sqrt(NUM_META_EPISODES)))
    y_coordinates = np.linspace(-3.0, 3.0, num=int(np.sqrt(NUM_META_EPISODES)))

    # map
    X, Y = np.meshgrid(x_coordinates, y_coordinates)
    Z = np.zeros((int(np.sqrt(NUM_META_EPISODES)), int(np.sqrt(NUM_META_EPISODES))))

    tasks = list()
    results = dict()

    for i in range(len(x_coordinates)):
        for j in range(len(y_coordinates)):
            tasks.append({
                "x_position": x_coordinates[i],
                "y_position": y_coordinates[j]
            })

            # index y.
            results[x_coordinates[i]] = list()
            continue

    X_list = list()
    Y_list = list()
    Z_list = list()
    with torch.no_grad():
        for iters in range(NUM_META_EPISODES // NUM_PROCESSES):
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
                            "r": info["meta_episode"]["r"]
                        })

                        # pass
                        x_position = info["meta_episode"]["sampled_task"][0]
                        y_position = info["meta_episode"]["sampled_task"][1]

                        # positions
                        X_list.append(x_position)
                        Y_list.append(y_position)
                        Z_list.append(info["meta_episode"]["r"])

                        idx = np.argwhere((X == x_position) & (Y == y_position))[0]
                        Z[idx[0]][idx[1]] = info["meta_episode"]["r"]
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
    with open('notebooks/heatmap-results.csv', 'w', newline = '') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(meta_episode_results)
        pass

    # contours
    plt.contour(X, Y, Z, 20, cmap = 'RdGy')

    # df
    data = pd.DataFrame(data = {'x': X_list, 'y': Y_list, 'z': Z_list})
    data = data.pivot(index = 'x', columns = 'y', values = 'z')
    sns.heatmap(data)
    plt.show()
    pass
