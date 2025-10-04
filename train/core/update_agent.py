import train.mytypes as train_types
from agents import BaseAgent

from typing import Any, Tuple
from functools import partial

import jax
import chex
import jax.numpy as jnp
from flax import nnx

@chex.dataclass
class UpdateState:
    agent: BaseAgent
    optimizer: nnx.Optimizer
    metrics: nnx.MultiMetric
    key: chex.PRNGKey

@partial(nnx.jit, static_argnames=('num_minibatches', 'num_ppo_epoch'))
def update_agent(
    agent: BaseAgent,
    mag_agent: BaseAgent,
    optimizer: nnx.Optimizer,
    dataset: train_types.Dataset,  # shape (batch_size, ...)
    metrics: nnx.MultiMetric,
    key: chex.PRNGKey,
    ent_coef: float,
    mag_coef: float,
    clip_eps: float,
    num_minibatches: int,
    num_ppo_epoch: int,
) -> Tuple[BaseAgent, nnx.Optimizer, nnx.MultiMetric]:
    """
    Updates agent parameters using PPO with optional magnetic regularization.

    All agents share the same model and all samples are used for training.

    Args:
        agent: Agent to update
        mag_agent: Optional magnetic agent for regularization
        optimizer: Optimizer state
        dataset: Training dataset with advantages and targets, shape (batch_size,)
        metrics: Metrics collector
        key: Random key for shuffling
        ent_coef: Entropy regularization coefficient
        mag_coef: Magnetic regularization coefficient
        clip_eps: PPO clipping parameter
        num_minibatches: Number of minibatches per epoch
        num_ppo_epoch: Number of training epochs
    Returns:
        Tuple of (updated_agent, updated_optimizer, updated_metrics)
    """
    batch_size = dataset.advantage.shape[0]

    assert batch_size % num_minibatches == 0, f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})"
    

    def calculate_n_log_loss(
        agent: BaseAgent, dataset: train_types.Dataset, metrics: nnx.MultiMetric
    ) -> chex.Numeric:
        """calculate loss and log to metrics"""
        dists = agent.get_action_distribution(dataset.observation, dataset.action_mask)

        """actor loss"""
        log_prob = dists.log_prob(dataset.action)

        # normalize advantage
        advantage_mean = jnp.mean(dataset.advantage)
        advantage_std = jnp.std(dataset.advantage)
        dataset.advantage = (dataset.advantage - advantage_mean) / (advantage_std + 1e-8)

        # ppo loss
        log_ratio = log_prob - dataset.log_prob
        ratio = jnp.exp(log_ratio)
        ppo_loss1 = ratio * dataset.advantage
        ppo_loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * dataset.advantage
        ppo_loss = -jnp.mean(jnp.minimum(ppo_loss1, ppo_loss2))

        # entropy loss
        entropy_loss = -jnp.mean(dists.entropy())

        # magnet loss
        mag_loss, mag_kl = 0, 0
        if mag_agent is not None:
            mag_dists = mag_agent.get_action_distribution(dataset.observation, dataset.action_mask)
            mag_kl = jnp.mean(dists.kl_divergence(mag_dists))
            mag_loss = mag_kl

        # total actor loss
        actor_loss = ppo_loss + ent_coef * entropy_loss + mag_coef * mag_loss

        """critic loss"""
        values = agent.get_value(dataset.observation)
        values_clipped = dataset.value + jnp.clip(values - dataset.value, -clip_eps, clip_eps)
        critic_loss1 = jnp.square(values - dataset.target_value)
        critic_loss2 = jnp.square(values_clipped - dataset.target_value)
        critic_loss = 0.5 * jnp.mean(jnp.maximum(critic_loss1, critic_loss2))

        """logging"""
        total_loss = actor_loss + critic_loss
        approx_kl = jnp.mean((ratio - 1) - log_ratio)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype('float32'))

        # explained variance calculation
        target_var = jnp.var(dataset.target_value)
        residual_var = jnp.var(dataset.target_value - values)
        explained_var = jnp.maximum(1 - residual_var / (target_var + 1e-8), jnp.float32(0))

        metrics.update(
            actor_loss = actor_loss,
            ppo_loss = ppo_loss,
            critic_loss=critic_loss,
            entropy = -entropy_loss,
            mag_kl = mag_kl,
            approx_kl = approx_kl,
            clip_frac = clip_frac,
            explained_var = explained_var
        )

        return total_loss


    def update_batch(carry: UpdateState, batch: train_types.Dataset):
        """Update the agent for a single batch"""
        # compute the gradient
        grad = nnx.grad(calculate_n_log_loss)(carry.agent, batch, carry.metrics)

        # update agent, optimizer state (inplace update)
        carry.optimizer.update(carry.agent, grad)

        return carry, 0


    def update_epoch(carry: UpdateState, _: Any):
        """Update the agent for a single epoch"""
        carry.key, shuffle_key1 = jax.random.split(carry.key, 2)

        # Shuffle data and create minibatches
        permutation1 = jax.random.permutation(shuffle_key1, batch_size)
        def process_batch1(x: chex.Array):
            # shuffle
            x = jnp.take(x, permutation1, axis=0)
            # create mini-batches
            x = jnp.reshape(x, (num_minibatches, -1, *x.shape[1:]))
            return x
        batched_dataset = jax.tree.map(process_batch1, dataset) # (num_minibatches, batch_size, ...)

        # update batches
        carry, _ = nnx.scan(update_batch)(carry, batched_dataset)

        return carry, 0


    # create update state for carrying
    carry = UpdateState(
        agent=agent,
        optimizer=optimizer,
        metrics=metrics,
        key=key
    )

    # perform ppo update for given epoch
    carry, _ = nnx.scan(update_epoch, length=num_ppo_epoch)(carry, None)
    carry: UpdateState = carry # for type hint
    

    return carry.agent, carry.optimizer, carry.metrics