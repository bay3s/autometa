from dataclasses import dataclass, fields, asdict
import json


@dataclass
class BaseTrainingConfig:
    """
    Base dataclass to keep track of experiment configs.

    Params:
      algo (str): Training algo (eg. PPO).
      env_name (str): Environment to use for training.
      env_configs (dict): Additional configs for each of the meta-environments.
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
    """

    # algo
    algo: str

    # env
    env_id: str
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
    pass

    @classmethod
    def from_json(cls, json_file_path: str) -> "BaseTrainingConfig":
        """
        Takes the json file path as parameter and returns the populated `BaseTrainingConfig`.

        Args:
            json_file_path (str): JSON file path from which to load the configs.

        Returns:
            BaseTrainingConfig
        """
        keys = [f.name for f in fields(cls)]
        file = json.load(open(json_file_path))

        return cls(**{key: file[key] for key in keys if key in file})

    def to_json(self) -> str:
        """
        Return JSON string with dataclass fields.

        Returns:
            str
        """
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_dict(cls, state_dict: str) -> "BaseTrainingConfig":
        """
        Takes the state `dict` as parameter and returns the populated TrainingConfigs.

        Args:
            state_dict (str): State dict from which to load the configs.

        Returns:
            BaseTrainingConfig
        """
        keys = [f.name for f in fields(cls)]

        return cls(**{key: state_dict[key] for key in keys if key in state_dict})

    def to_dict(self) -> dict:
        """
        Return dictionary with dataclass fields.

        Returns:
            dict
        """
        return {k: str(v) for k, v in asdict(self).items()}
