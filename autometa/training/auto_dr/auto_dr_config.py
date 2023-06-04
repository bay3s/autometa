from dataclasses import dataclass

from autometa.training.base_training_config import BaseTrainingConfig


@dataclass
class AutoDRConfig(BaseTrainingConfig):
    """
    Dataclass to keep track of experiment configs.

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
      adr_evaluation_probability (float): Probability of evaluating the training performance.
      adr_performance_buffer_size (int): Buffer size used for evaluation.
      adr_performance_threshold_lower (int): Lower threshold for performance, if this is not met entropy is reduced.
      adr_performance_threshold_upper (int): Upper threshold for performance, if this is met entropy is increased.
      adr_delta (float): Delta value for auto-dr.
    """

    # adr
    adr_evaluation_probability: float
    adr_performance_buffer_size: int
    adr_performance_threshold_lower: int
    adr_performance_threshold_upper: int
    adr_delta: float
    pass
