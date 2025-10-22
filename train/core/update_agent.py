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

@partial(nnx.jit, static_argnames=('num_minibatches', 'num_ppo_epoch', 'normalize_logprob'))
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
    normalize_logprob: bool = True,
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
        normalize_logprob: If True, normalize log_prob and KL divergence by number of action dimensions
                          (important for multi-discrete actions to prevent instability)
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

        # Extract validity mask and compute number of valid samples
        valid_mask = dataset.valid_mask.astype(jnp.float32)  # Convert bool to float for masking
        num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)  # Avoid division by zero

        """actor loss"""
        log_prob = dists.log_prob(dataset.action)

        # Compute normalization factor based on action dimensions
        # For discrete: action.ndim = 1, n_action_dims = 1
        # For multi-discrete: action.ndim = 2, n_action_dims = action.shape[-1]
        n_action_dims = jnp.where(
            dataset.action.ndim == 1,
            1,  # discrete action
            dataset.action.shape[-1]  # multi-discrete action
        )
        # If normalize_logprob=False, use 1.0; otherwise use n_action_dims
        norm_factor = jnp.where(normalize_logprob, jnp.float32(n_action_dims), jnp.float32(1.0))

        # Normalize log_prob by action dimensions (always divide, factor is 1.0 if disabled)
        log_prob_normalized = log_prob / norm_factor
        old_log_prob_normalized = dataset.log_prob / norm_factor

        # normalize advantage (only over valid samples)
        masked_advantage = dataset.advantage * valid_mask
        advantage_mean = jnp.sum(masked_advantage) / num_valid
        advantage_var = jnp.sum(valid_mask * jnp.square(dataset.advantage - advantage_mean)) / num_valid
        advantage_std = jnp.sqrt(advantage_var)
        dataset.advantage = (dataset.advantage - advantage_mean) / (advantage_std + 1e-8)

        # ppo loss (use normalized log_prob, masked mean)
        log_ratio = log_prob_normalized - old_log_prob_normalized
        ratio = jnp.exp(log_ratio)
        ppo_loss1 = ratio * dataset.advantage
        ppo_loss2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * dataset.advantage
        ppo_loss_per_sample = -jnp.minimum(ppo_loss1, ppo_loss2)
        ppo_loss = jnp.sum(ppo_loss_per_sample * valid_mask) / num_valid

        # entropy loss (masked mean)
        entropy_per_sample = dists.entropy()
        entropy_loss = -jnp.sum(entropy_per_sample * valid_mask) / num_valid

        # magnet loss (masked mean)
        mag_loss, mag_kl = 0, 0
        if mag_agent is not None:
            mag_dists = mag_agent.get_action_distribution(dataset.observation, dataset.action_mask)
            kl_div = dists.kl_divergence(mag_dists)
            # Normalize KL divergence by action dimensions
            kl_div_normalized = kl_div / norm_factor
            mag_kl = jnp.sum(kl_div_normalized * valid_mask) / num_valid
            mag_loss = mag_kl

        # total actor loss
        actor_loss = ppo_loss + ent_coef * entropy_loss + mag_coef * mag_loss

        """critic loss (masked mean)"""
        values = agent.get_value(dataset.observation)
        values_clipped = dataset.value + jnp.clip(values - dataset.value, -clip_eps, clip_eps)
        critic_loss1 = jnp.square(values - dataset.target_value)
        critic_loss2 = jnp.square(values_clipped - dataset.target_value)
        critic_loss_per_sample = 0.5 * jnp.maximum(critic_loss1, critic_loss2)
        critic_loss = jnp.sum(critic_loss_per_sample * valid_mask) / num_valid

        """logging (all metrics computed only over valid samples)"""
        total_loss = actor_loss + critic_loss
        approx_kl = jnp.sum(((ratio - 1) - log_ratio) * valid_mask) / num_valid
        clip_frac = jnp.sum((jnp.abs(ratio - 1.0) > clip_eps).astype('float32') * valid_mask) / num_valid

        # explained variance calculation (only over valid samples)
        masked_target = dataset.target_value * valid_mask
        masked_values = values * valid_mask
        target_mean = jnp.sum(masked_target) / num_valid
        target_var = jnp.sum(valid_mask * jnp.square(dataset.target_value - target_mean)) / num_valid
        residual_var = jnp.sum(valid_mask * jnp.square(dataset.target_value - values)) / num_valid
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