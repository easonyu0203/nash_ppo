import train.mytypes as train_types
from agents import BaseAgent

from typing import Any, Literal, Tuple
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

@partial(nnx.jit, static_argnames=('num_minibatches', 'num_ppo_epoch', 'only_use_player0_experience'))
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
    only_use_player0_experience: bool,
) -> Tuple[BaseAgent, nnx.Optimizer, nnx.MultiMetric]:
    """
    Updates agent parameters using PPO with optional magnetic regularization.
    
    Args:
        agent: Agent to update
        optimizer: Optimizer state
        dataset: Training dataset with advantages and targets, shape (batch_size,)
        metrics: Metrics collector
        key: Random key for shuffling
        ent_coef: Entropy regularization coefficient
        mag_coef: Magnetic regularization coefficient  
        clip_eps: PPO clipping parameter
        num_minibatches: Number of minibatches per epoch
        num_ppo_epoch: Number of training epochs
        only_use_player0_experience: If True, only train on player 0 data
        mag_agent: Optional magnetic agent for regularization
    Returns:
        Tuple of (updated_agent, updated_optimizer, updated_metrics)
    """
    batch_size = dataset.advantage.shape[0]

    assert batch_size % num_minibatches == 0, f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})"

    # mask the dataset if specified
    if only_use_player0_experience:
        # only use data where it is valid and is act by player 0
        dataset.valid_mask = jnp.logical_and(dataset.valid_mask, dataset.current_player == jnp.int32(0))

    def masked_mean(x: chex.Array, mask: chex.Array) -> chex.Numeric:
        """Compute mean only over valid (masked) samples"""
        masked_x = x * mask
        return jnp.sum(masked_x) / jnp.maximum(jnp.sum(mask), 1.0)
    
    def masked_var(x: chex.Array, mask: chex.Array) -> chex.Numeric:
        """Compute variance only over valid (masked) samples"""
        masked_mean_val = masked_mean(x, mask)
        masked_variance = jnp.sum(mask * jnp.square(x - masked_mean_val)) / jnp.maximum(jnp.sum(mask), 1.0)
        return masked_variance
    
    def masked_std(x: chex.Array, mask: chex.Array) -> chex.Numeric:
        """Compute standard deviation only over valid (masked) samples"""
        return jnp.sqrt(masked_var(x, mask))
    

    def calculate_n_log_loss(
        agent: BaseAgent, dataset: train_types.Dataset, metrics: nnx.MultiMetric
    ) -> chex.Numeric:
        """calculate loss and log to metrics"""
        dists = agent.get_action_distribution(dataset.observation, dataset.action_mask)
        
        """actor loss"""
        log_prob = dists.log_prob(dataset.action)
        
        # normalize advantage using valid mask
        advantage_mean = masked_mean(dataset.advantage, dataset.valid_mask)
        advantage_std = masked_std(dataset.advantage, dataset.valid_mask)
        dataset.advantage = (dataset.advantage - advantage_mean) / (advantage_std + 1e-8)

        # ppo loss
        log_ratio = log_prob - dataset.log_prob
        ratio = jnp.exp(log_ratio)
        ppo_loss1 = ratio * dataset.advantage
        ppo_loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * dataset.advantage
        ppo_loss = -masked_mean(jnp.minimum(ppo_loss1, ppo_loss2), dataset.valid_mask)

        # entropy loss
        entropy_loss = -masked_mean(dists.entropy(), dataset.valid_mask)

        # magnet loss
        mag_loss, mag_kl = 0, 0
        if mag_agent is not None:
            mag_dists = mag_agent.get_action_distribution(dataset.observation, dataset.action_mask)
            mag_kl = masked_mean(dists.kl_divergence(mag_dists), dataset.valid_mask)
            mag_loss = mag_kl

        # total actor loss
        actor_loss = ppo_loss + ent_coef * entropy_loss + mag_coef * mag_loss

        """critic loss"""
        values = agent.get_value(dataset.observation)
        values_clipped = dataset.value + jnp.clip(values - dataset.value, -clip_eps, clip_eps)
        critic_loss1 = jnp.square(values - dataset.target_value)
        critic_loss2 = jnp.square(values_clipped - dataset.target_value)
        critic_loss = 0.5 * masked_mean(jnp.maximum(critic_loss1, critic_loss2), dataset.valid_mask)

        """logging"""
        total_loss = actor_loss + critic_loss
        approx_kl = masked_mean((ratio - 1) - log_ratio, dataset.valid_mask)
        clip_frac = masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype('float32'), dataset.valid_mask)
        
        # explained variance calculation with masking
        target_var = masked_var(dataset.target_value, dataset.valid_mask)
        residual_var = masked_var(dataset.target_value - values, dataset.valid_mask)
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
        carry.optimizer.update(model=carry.agent, grads=grad)

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