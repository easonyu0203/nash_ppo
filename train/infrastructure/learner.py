"""
Setup utilities for training components initialization.

This module contains factory functions and helpers for initializing
training components like learner state, buffers, metrics, etc.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
import chex
import optax
from omegaconf import DictConfig
from gymnasium.spaces import Dict as DictSpace, Discrete, MultiDiscrete, Box

import envs.mytypes as env_types
from agents import create_agent, BaseAgent
from train.data import RolloutBuffer

if TYPE_CHECKING:
    from train.algorithms import ValueNorm


@dataclass
class LearnerState:
    """State container for the learning process."""
    key: chex.PRNGKey
    last_timestep: env_types.TimeStep
    agent: BaseAgent
    optimizer: nnx.Optimizer
    train_metrics: nnx.MultiMetric
    rollout_metrics: nnx.MultiMetric
    mag_agent: Optional[BaseAgent]  # use for regularization
    value_normalizer: Optional['ValueNorm']  # use for value normalization


def infer_buffer_shapes(env: env_types.BaseEnv) -> Tuple:
    """
    Infer observation and action shapes from environment spaces.

    Args:
        env: Environment instance

    Returns:
        Tuple of (obs_shape, action_shape) compatible with RolloutBuffer
    """
    # Handle Dict observation spaces
    if isinstance(env.observation_space, DictSpace):
        obs_shape = {key: space.shape for key, space in env.observation_space.spaces.items()}
    else:
        obs_shape = env.observation_space.shape

    # Handle different action space types
    if isinstance(env.action_space, Discrete):
        action_shape = ()
    elif isinstance(env.action_space, MultiDiscrete):
        action_shape = env.action_space.nvec.shape
    elif isinstance(env.action_space, Box):
        action_shape = env.action_space.shape
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    return obs_shape, action_shape


def create_rollout_buffer(
    env: env_types.BaseEnv,
    num_envs: int,
    num_steps: int
) -> RolloutBuffer:
    """
    Create a rollout buffer with appropriate shapes for the environment.

    Args:
        env: Environment instance
        num_envs: Number of parallel environments
        num_steps: Number of steps per rollout

    Returns:
        Initialized RolloutBuffer
    """
    obs_shape, action_shape = infer_buffer_shapes(env)

    return RolloutBuffer(
        num_envs=num_envs,
        num_steps=num_steps,
        num_agents=env.num_agents,
        obs_shape=obs_shape,
        action_shape=action_shape
    )


def create_training_metrics() -> nnx.MultiMetric:
    """
    Create metrics tracker for training statistics.

    Returns:
        Initialized MultiMetric for training
    """
    return nnx.MultiMetric(
        actor_loss=nnx.metrics.Average("actor_loss"),
        ppo_loss=nnx.metrics.Average("ppo_loss"),
        entropy=nnx.metrics.Average("entropy"),
        critic_loss=nnx.metrics.Average("critic_loss"),
        approx_kl=nnx.metrics.Average("approx_kl"),
        mag_kl=nnx.metrics.Average("mag_kl"),
        clip_frac=nnx.metrics.Average("clip_frac"),
        explained_var=nnx.metrics.Average("explained_var"),
    )


def create_rollout_metrics() -> nnx.MultiMetric:
    """
    Create metrics tracker for rollout statistics.

    Returns:
        Initialized MultiMetric for rollouts
    """
    return nnx.MultiMetric(
        inverse_eps_len=nnx.metrics.Average("inverse_eps_len"),
        reward=nnx.metrics.Average("reward"),
    )


def create_learner_state(
    config: DictConfig,
    init_timestep: env_types.TimeStep,
    key: chex.PRNGKey
) -> LearnerState:
    """
    Initialize learner state with all required components.

    Args:
        config: Training configuration
        init_timestep: Initial environment timestep
        key: Random key for initialization

    Returns:
        Initialized LearnerState
    """
    # Create agent
    key, agent_key = jax.random.split(key)
    agent = create_agent(config.agent, key=agent_key)

    # Create optimizer
    optimizer = nnx.Optimizer(
        agent,
        optax.adamw(config.algorithm.lr, eps=1e-5),
        wrt=nnx.Param
    )

    # Create metrics
    train_metrics = create_training_metrics()
    rollout_metrics = create_rollout_metrics()

    # Create value normalizer if enabled
    value_normalizer = None
    if config.algorithm.normalize_value:
        from train.algorithms import ValueNorm
        value_normalizer = ValueNorm()

    # Create learner state
    key, learner_key = jax.random.split(key)
    return LearnerState(
        key=learner_key,
        last_timestep=init_timestep,
        agent=agent,
        optimizer=optimizer,
        train_metrics=train_metrics,
        rollout_metrics=rollout_metrics,
        mag_agent=nnx.clone(agent),  # init as the same
        value_normalizer=value_normalizer,
    )


def process_rollout_metrics(rollout_metrics: dict) -> dict:
    """
    Process raw rollout metrics into meaningful statistics.

    Args:
        rollout_metrics: Raw metrics dict from MultiMetric.compute()

    Returns:
        Processed metrics dict with episode length and return
    """
    eps_len = 1 / rollout_metrics['inverse_eps_len']
    ret = rollout_metrics['reward'] / rollout_metrics['inverse_eps_len']

    return {
        'eps_len': eps_len,
        'return': ret
    }


def create_value_norm_metrics(value_normalizer: 'ValueNorm') -> dict:
    """
    Extract value normalization statistics for logging.

    Args:
        value_normalizer: ValueNorm instance

    Returns:
        Dict of value normalization metrics
    """
    mean, var = value_normalizer.running_mean_var()
    return {
        'value_norm/mean': mean[0],  # Extract scalar from shape (1,)
        'value_norm/std': jnp.sqrt(var)[0],
        'value_norm/debiasing_term': value_normalizer.debiasing_term.value
    }
