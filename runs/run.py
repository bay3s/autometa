import argparse

from autometa.utils.path_utils import absolute_path
from autometa.training.rl_squared.rl_squared_config import RLSquaredConfig
from autometa.training.rl_squared.rl_squared_trainer import RLSquaredTrainer

from autometa.training.auto_dr.auto_dr_config import AutoDRConfig
from autometa.training.auto_dr.auto_dr_trainer import AutoDRTrainer
from autometa.utils.env_utils import register_custom_envs

register_custom_envs()


POINT_NAVIGATION = "point_navigation"
ANT_VELOCITY = "ant_velocity"
ANT_NAVIGATION = "ant_navigation"
CHEETAH_VELOCITY = "cheetah_velocity"
WALKER_DYNAMICS = "walker_dynamics"
HOPPER_DYNAMICS = "hopper_dynamics"

SUPPORTED_ENVIRONMENTS = [
    POINT_NAVIGATION,
    CHEETAH_VELOCITY,
    ANT_VELOCITY,
    ANT_NAVIGATION,
    WALKER_DYNAMICS,
    HOPPER_DYNAMICS,
]

RL_SQUARED = "rl_squared"
AUTO_DR = "auto_dr"

SUPPORTED_ALGOS = [RL_SQUARED, AUTO_DR]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="autometa",
        description="Script to run experiments on various meta-learning benchmarks.",
    )

    parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        default=None,
        help=f"Training algorithm, one of [{', '.join(SUPPORTED_ALGOS)}].",
    )

    parser.add_argument(
        "--env",
        choices=SUPPORTED_ENVIRONMENTS,
        default=None,
        help=f"Training environment, one of [{', '.join(SUPPORTED_ENVIRONMENTS)}].",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Absolute checkpoint path, if any, from which to restart the training run.",
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

    parser.add_argument(
        "--checkpoint-all",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Whether to save all the checkpoints or just the last one.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        action=argparse.BooleanOptionalAction,
        help=f"Iteration interval between two consecutive checkpoints.",
    )

    # args
    args = parser.parse_args()

    if args.env is None and not args.run_all:
        raise ValueError(
            f"Unable to infer experiment environment from the inputs, either provide `--env-name` or "
            f"set `--run-all` to `True`"
        )

    if args.algo is None or args.algo not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unable to infer algorithm from the inputs, either `rl_squared` or `auto-dr`"
        )

    # config
    config_path = absolute_path(f"runs/configs/{args.algo}/{args.algo}_{args.env}.json")
    config_cls = RLSquaredConfig if args.algo == RL_SQUARED else AutoDRConfig
    training_config = config_cls.from_json(config_path)

    # trainer
    trainer_cls = RLSquaredTrainer if args.algo == RL_SQUARED else AutoDRTrainer
    trainer = trainer_cls(config=training_config, checkpoint_path=args.checkpoint)

    # train
    trainer.train(
        enable_wandb=not args.disable_wandb,
        is_dev=not args.prod,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_all=args.checkpoint_all,
    )
    pass
