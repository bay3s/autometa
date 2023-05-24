import os

import argparse

from autometa.training.trainer import ExperimentConfig
from autometa.training.trainer import Trainer
from autometa.utils.env_utils import register_custom_envs

register_custom_envs()


POINT_ROBOT_NAVIGATION_ENV = "point_robot_navigation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RL-Squared",
        description="Script to run experiments on various meta-learning benchmarks.",
    )

    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint, if any, from which to restart the training run.",
    )

    parser.add_argument(
        "--disable-wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether to log the experiment to `wandb`.",
    )

    parser.add_argument(
        "--prod",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether this a production run of the experiment.",
    )

    # args
    args = parser.parse_args()

    # config
    config_path = f"{os.path.dirname(__file__)}/configs/{POINT_ROBOT_NAVIGATION_ENV}.json"
    experiment_config = ExperimentConfig.from_json(config_path)

    # train
    trainer = Trainer(experiment_config, restart_checkpoint=args.from_checkpoint)
    trainer.train(enable_wandb=not args.disable_wandb, is_dev=not args.prod)
    pass
