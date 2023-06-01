from dataclasses import dataclass, fields, asdict
import os
import json
from datetime import datetime


@dataclass
class BaseConfig:
    """
    Base dataclass to keep track of experiment configs.

    Params:
      algo (str): Algo to train.
      env_name (str): Environment to use for training.
      env_configs (dict): Additional configs for each of the meta-environments.
      normalize_obs (bool): Whether to normalize observations.
      normalize_rew (bool): Whether to normalize observations.
      max_policy_iterations (int): Number of total steps to train over.
      actor_lr (float): Learning rate of the actor.
      critic_lr (float): Learning rate of the critic / value function.
      optimizer_eps (float): `eps` parameter value for Adam or RMSProp
      max_grad_norm (float): Max grad norm for gradient clipping.
      use_linear_lr_decay (bool): Whether to use linear learning rate decay in training.
      random_seed (int): Random seed.
      no_cuda (bool): Whether to avoid using CUDA even if a GPU is available.
      cuda_deterministic (float): Whether to use a deterministic version of CUDA.
      steps_per_trial (int): Number of steps per RL-Squared trial (one trial includes multiple episodes).
      num_processes (int): Number of parallel training processes.
      discount_gamma (float): Discount applied to trajectories that are sampled.
      ppo_epochs (int): Number of PPO epochs for training.
      ppo_clip_param (int): The `epsilon` clip parameter for the surrogate objective.
      ppo_entropy_coef (float): Entropy coefficient.
      ppo_value_loss_coef (float): Value loss coefficient.
      ppo_num_minibatches (int): Number of minibatches for PPO
      use_gae (bool): Whether to use generalized advantage estimates.
      gae_lambda (float): Lambda parameter for GAE.
      log_interval (int): Interval between logging.
      log_dir (str): Directory to log to.
      checkpoint_interval (int): Number of updates between each checkpoint.
      checkpoint_all (bool): Whether to checkpoint all models or just the last one.
    """

    # algo
    algo: str

    # env
    env_name: str
    env_configs: dict
    norm_observations: bool
    norm_rewards: bool

    # opt / grad clipping
    use_linear_lr_decay: bool
    actor_lr: float
    critic_lr: float
    optimizer_eps: float
    max_grad_norm: float

    # setup
    random_seed: int
    cuda_deterministic: float
    use_cuda: bool

    # sampling
    policy_iterations: int
    meta_episodes_per_epoch: int
    meta_episode_length: int
    num_processes: int
    discount_gamma: float

    # ppo
    ppo_opt_epochs: int
    ppo_clip_param: float
    ppo_entropy_coef: float
    ppo_value_loss_coef: float
    ppo_num_minibatches: int

    # advantage
    use_gae: bool
    gae_lambda: bool

    # checkpointing
    checkpoint_interval: int
    checkpoint_all: bool
    pass

    def __post_init__(self):
        """
        Sets a timestamp for the config.

        Returns:
            None
        """
        self._timestamp = int(datetime.timestamp(datetime.now()))
        self._wandb_run_id = None
        pass

    @property
    def run_id(self) -> str:
        """
        Returns a timestamp for the experiment config

        Returns:
            str
        """
        if self._wandb_run_id is not None:
            return self._wandb_run_id

        return str(self._timestamp)

    @run_id.setter
    def run_id(self, wandb_run_id: str) -> str:
        """
        Returns a timestamp for the experiment config

        Args:
            wandb_run_id (str): `wandb` run id.

        Returns:
            str
        """
        self._wandb_run_id = wandb_run_id

    @property
    def directory(self) -> str:
        """
        Return the directory to store logs.

        Returns:
          str
        """
        folder = self.env_name

        for pos, ch in enumerate(folder):
            if ch.isupper() and pos > 0:
                folder = folder.replace(ch, "-%s" % ch.lower())
            elif ch.isupper():
                folder = folder.replace(ch, ch.lower())

        return f"./results/{self.algo}/{folder}/run-{self.run_id}/"

    @property
    def log_dir(self) -> str:
        """
        Return the directory to store logs.

        Returns:
            str
        """
        return f"{self.directory}/logs/"

    @property
    def checkpoint_dir(self) -> str:
        """
        Returns the directory to store checkpoints.

        Returns:
            str
        """
        return f"{self.directory}/checkpoints/"

    @classmethod
    def from_json(cls, json_file_path: str) -> "BaseConfig":
        """
        Takes the json file path as parameter and returns the populated TrainingConfigs.

        Returns:
            BaseConfig
        """
        keys = [f.name for f in fields(cls)]
        file = json.load(open(json_file_path))

        return cls(**{key: file[key] for key in keys if key in file})

    @property
    def json(self) -> str:
        """
        Return JSON string with dataclass fields.

        Returns:
            str
        """
        return json.dumps(self.__dict__, indent=2)

    @property
    def dict(self) -> dict:
        """
        Return dictionary with dataclass fields.

        Returns:
            dict
        """
        return {k: str(v) for k, v in asdict(self).items()}

    def save(self) -> None:
        """
        Returns the checkpoint directory.

        Returns:
            str
        """
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        with open(f"{self.directory}/config.json", "w") as outfile:
            outfile.write(self.json)
            pass
