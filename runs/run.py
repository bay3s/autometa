import os

import argparse

from autometa.training.rl_squared.rl_squared_config import RLSquaredConfig
from autometa.training.rl_squared.rl_squared_trainer import RLSquaredTrainer

from autometa.training.auto_dr.auto_dr_config import AutoDRConfig
from autometa.training.auto_dr.auto_dr_trainer import AutoDRTrainer

from autometa.utils.env_utils import register_custom_envs

register_custom_envs()


POINT_ROBOT_NAV = "point_robot_navigation"

SUPPORTED_ENVIRONMENTS = [
    POINT_ROBOT_NAV,
]

RL_SQUARED = "rl_squared"
AUTO_DR = "auto_dr"

SUPPORTED_ALGOS = [
    RL_SQUARED,
    AUTO_DR
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="AutoMeta",
        description="Script to run experiments on various meta-learning benchmarks.",
    )

    parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        default=None,
        help=f"Training algorithm, one of [{', '.join(SUPPORTED_ALGOS)}].",
    )

    parser.add_argument(
        "--env-name",
        choices=SUPPORTED_ENVIRONMENTS,
        default=None,
        help=f"Training environment, one of [{', '.join(SUPPORTED_ENVIRONMENTS)}].",
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

    if args.env_name is None and not args.run_all:
        raise ValueError(
            f"Unable to infer experiment environment from the inputs, either provide `--env-name` or "
            f"set `--run-all` to `True`"
        )

    if args.algo is None or args.algo not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unable to infer algorithm from the inputs, either `rl_squared` or `auto-dr`"
        )

    # config
    config_path = (
        f"{os.path.dirname(__file__)}/configs/{args.algo}/{args.env_name}.json"
    )

    # train
    if args.algo == RL_SQUARED:
        experiment_config = RLSquaredConfig.from_json(config_path)
        trainer = RLSquaredTrainer(experiment_config, restart_checkpoint=args.from_checkpoint)
        trainer.train(enable_wandb=not args.disable_wandb, is_dev=not args.prod)
    elif args.algo == AUTO_DR:
        experiment_config = AutoDRConfig.from_json(config_path)
        # @todo add checkpoint
        trainer = AutoDRTrainer(experiment_config)
        trainer.train(enable_wandb = not args.disable_wandb, is_dev = not args.prod)
        pass
