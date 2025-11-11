"""PPO algorithm implementation with support for both stateless and stateful agents."""

from typing import Tuple, Optional, Any
from functools import partial

import jax
import jax.numpy as jnp
import chex
from flax import nnx
import optax

import train.types as train_types
from agents import BaseAgent, StatefulAgent
from train.algorithms.value_norm import ValueNorm
from train.algorithms.ppo_stateless import calculate_loss_stateless
from train.algorithms.ppo_stateful import calculate_loss_stateful


@chex.dataclass
class UpdateState:
    """State carrier for PPO updates."""
    agent: BaseAgent
    optimizer: nnx.Optimizer
    metrics: nnx.MultiMetric
    key: chex.PRNGKey


@partial(nnx.jit, static_argnames=('num_minibatches', 'num_ppo_epoch', 'normalize_logprob'))
def update_agent(
    agent: BaseAgent,
    mag_agent: BaseAgent,
    optimizer: nnx.Optimizer,
    dataset: train_types.Dataset,
    metrics: nnx.MultiMetric,
    key: chex.PRNGKey,
    ent_coef: float,
    mag_coef: float,
    clip_eps: float,
    num_minibatches: int,
    num_ppo_epoch: int,
    normalize_logprob: bool = True,
    value_normalizer: Optional[ValueNorm] = None,
) -> Tuple[BaseAgent, nnx.Optimizer, nnx.MultiMetric]:
    """Update agent parameters using PPO.

    Supports both stateless agents (flat batches) and stateful agents (BPTT sequences).
    Automatically dispatches to appropriate loss function based on agent type.

    Args:
        agent: Agent to update
        mag_agent: Magnetic agent for regularization (can be None)
        optimizer: Optimizer state
        dataset: Training dataset
                 - Stateless: shape (batch_size, ...)
                 - Stateful: shape (batch_size, bptt_length, ...)
        metrics: Metrics collector
        key: Random key for shuffling
        ent_coef: Entropy regularization coefficient
        mag_coef: Magnetic regularization coefficient
        clip_eps: PPO clipping parameter
        num_minibatches: Number of minibatches per epoch
        num_ppo_epoch: Number of training epochs
        normalize_logprob: If True, normalize log_prob by action dimensions
        value_normalizer: Optional value normalizer

    Returns:
        Tuple of (updated_agent, updated_optimizer, updated_metrics)
    """
    batch_size = dataset.advantage.shape[0]

    assert batch_size % num_minibatches == 0, \
        f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})"

    # Update value normalizer statistics and normalize targets BEFORE training loop
    if value_normalizer is not None:
        # Flatten target values for consistent shape with value normalizer
        # For stateless: already (batch_size,)
        # For stateful: (num_sequences, bptt_length) -> (batch_size,)
        target_value_flat = dataset.target_value.reshape(-1)
        value_normalizer.update(target_value_flat)
        normalized_flat = value_normalizer.normalize(target_value_flat)
        # Reshape back to original shape
        dataset = dataset.replace(target_value=normalized_flat.reshape(dataset.target_value.shape))

    # Choose loss function based on agent type
    is_stateful = isinstance(agent, StatefulAgent)

    def calculate_loss(agent: BaseAgent, dataset: train_types.Dataset) -> Tuple[chex.Numeric, dict[str, chex.Numeric]]:
        """Wrapper that dispatches to appropriate loss function.

        Returns:
            Tuple of (total_loss, aux_losses) where aux_losses contains metrics
        """
        if is_stateful:
            return calculate_loss_stateful(
                agent, mag_agent, dataset,
                ent_coef, mag_coef, clip_eps, normalize_logprob
            )
        else:
            return calculate_loss_stateless(
                agent, mag_agent, dataset,
                ent_coef, mag_coef, clip_eps, normalize_logprob
            )

    def update_batch(carry: UpdateState, batch: train_types.Dataset):
        """Update the agent for a single batch."""
        # Compute gradient with auxiliary outputs
        grad, aux_losses = nnx.grad(calculate_loss, has_aux=True)(carry.agent, batch)

        # Compute global gradient norm before clipping
        global_norm = optax.global_norm(grad)

        # Update metrics with auxiliary losses and gradient norm
        carry.metrics.update(
            grad_norm=global_norm,
            **aux_losses
        )

        # Update agent and optimizer state (in-place)
        carry.optimizer.update(carry.agent, grad)

        return carry, None

    def update_epoch(carry: UpdateState, _: Any):
        """Update the agent for a single epoch."""
        carry.key, shuffle_key = jax.random.split(carry.key, 2)

        # Shuffle data and create minibatches
        permutation = jax.random.permutation(shuffle_key, batch_size)

        def shuffle_and_minibatch(x: chex.Array):
            # Shuffle
            x = jnp.take(x, permutation, axis=0)
            # Create minibatches
            x = jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))
            return x

        batched_dataset = jax.tree.map(shuffle_and_minibatch, dataset)

        # Update batches
        carry, _ = nnx.scan(update_batch)(carry, batched_dataset)

        return carry, None

    # Create update state
    carry = UpdateState(
        agent=agent,
        optimizer=optimizer,
        metrics=metrics,
        key=key
    )

    # Perform PPO updates for specified number of epochs
    carry, _ = nnx.scan(update_epoch, length=num_ppo_epoch)(carry, None)

    return carry.agent, carry.optimizer, carry.metrics
