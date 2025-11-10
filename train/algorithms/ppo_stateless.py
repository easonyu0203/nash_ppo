"""PPO training for stateless agents (flat batches)."""

import jax.numpy as jnp
import chex
from flax import nnx

import train.infrastructure.types as train_types
from agents import BaseAgent
from train.algorithms.ppo_utils import get_action_norm_factor

# Constants
_EPSILON = 1e-8


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> chex.Numeric:
    """Compute mean of values where mask is 1, ignoring masked-out values."""
    num_valid = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(values * mask) / num_valid


def _normalize_advantages(
    advantages: jnp.ndarray,
    valid_mask: jnp.ndarray
) -> jnp.ndarray:
    """Normalize advantages using mean and std from valid samples only."""
    adv_mean = _masked_mean(advantages, valid_mask)
    adv_var = _masked_mean(jnp.square(advantages - adv_mean), valid_mask)
    adv_std = jnp.sqrt(adv_var)
    return (advantages - adv_mean) / (adv_std + _EPSILON)


def _compute_agent_outputs(
    agent: BaseAgent,
    mag_agent: BaseAgent | None,
    observations: chex.ArrayTree,
    actions: jnp.ndarray,
    norm_factor: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute agent outputs for the given observations and actions.

    Args:
        agent: Agent to evaluate
        mag_agent: Optional magnetic agent for KL computation
        observations: Batch of observations (can be pytree or array)
        actions: Batch of actions
        norm_factor: Normalization factor for log probs

    Returns:
        Tuple of (log_probs, entropies, values, kls) for the batch
    """
    # Get action distribution and value from agent simultaneously
    action_dist, values = agent.get_distribution_and_value(observations)

    # Compute policy outputs
    log_probs = action_dist.log_prob(actions) / norm_factor
    entropies = action_dist.entropy() / norm_factor

    # Compute KL divergence with magnetic agent if available
    kls = jnp.zeros_like(log_probs)
    if mag_agent is not None:
        mag_dist, _ = mag_agent.get_distribution_and_value(observations)
        kls = action_dist.kl_divergence(mag_dist) / norm_factor

    return log_probs, entropies, values, kls


def _compute_ppo_loss(
    log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    valid_mask: jnp.ndarray,
    clip_eps: float,
) -> tuple[chex.Numeric, chex.Numeric, chex.Numeric]:
    """Compute clipped PPO policy loss and diagnostic metrics.

    Returns:
        Tuple of (ppo_loss, approx_kl, clip_fraction)
    """
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    # Clipped surrogate objective
    unclipped_objective = ratio * advantages
    clipped_objective = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    ppo_loss = -_masked_mean(jnp.minimum(unclipped_objective, clipped_objective), valid_mask)

    # Diagnostic metrics
    approx_kl = _masked_mean((ratio - 1.0) - log_ratio, valid_mask)
    clip_frac = _masked_mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32), valid_mask)

    return ppo_loss, approx_kl, clip_frac


def _compute_value_loss(
    values: jnp.ndarray,
    old_values: jnp.ndarray,
    target_values: jnp.ndarray,
    valid_mask: jnp.ndarray,
    clip_eps: float,
) -> chex.Numeric:
    """Compute clipped value function loss."""
    # Clipped value targets
    values_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)

    # Clipped MSE loss
    unclipped_loss = jnp.square(values - target_values)
    clipped_loss = jnp.square(values_clipped - target_values)
    value_loss = 0.5 * _masked_mean(jnp.maximum(unclipped_loss, clipped_loss), valid_mask)

    return value_loss


def _compute_explained_variance(
    values: jnp.ndarray,
    target_values: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> chex.Numeric:
    """Compute explained variance metric for value function."""
    target_mean = _masked_mean(target_values, valid_mask)
    target_var = _masked_mean(jnp.square(target_values - target_mean), valid_mask)
    residual_var = _masked_mean(jnp.square(target_values - values), valid_mask)
    explained_var = 1.0 - residual_var / (target_var + _EPSILON)
    return jnp.maximum(explained_var, 0.0)


def calculate_loss_stateless(
    agent: BaseAgent,
    mag_agent: BaseAgent,
    dataset: train_types.Dataset,
    ent_coef: float,
    mag_coef: float,
    clip_eps: float,
    normalize_logprob: bool,
) -> tuple[chex.Numeric, dict[str, chex.Numeric]]:
    """Calculate PPO loss for stateless agents (flat batches).

    Dataset shape: (batch_size, ...)
    Each sample is a single timestep (no temporal dependencies).

    Args:
        agent: Agent to train
        mag_agent: Magnetic agent for regularization (can be None)
        dataset: Training dataset with flat batches
        ent_coef: Entropy coefficient
        mag_coef: Magnetic loss coefficient
        clip_eps: PPO clipping epsilon
        normalize_logprob: Whether to normalize log probs by action dimensions

    Returns:
        Tuple of (total_loss, aux_losses) where aux_losses is a dict containing:
        - actor_loss, ppo_loss, critic_loss, entropy, mag_kl, approx_kl, clip_frac, explained_var
    """
    # Compute valid_mask: True if transition is valid (state was NOT done)
    # When done=True, the action was taken in a terminal/truncated state and should be ignored
    valid_mask = (~dataset.done).astype(jnp.float32)

    # Get normalization factor for action space
    norm_factor = get_action_norm_factor(
        agent.action_dim,
        agent.action_space_type,
        normalize_logprob
    )

    # Normalize advantages across all valid samples
    normalized_advantages = _normalize_advantages(dataset.advantage, valid_mask)

    # ========== Agent Forward Pass ==========
    # Compute current policy and value outputs
    log_probs, entropies, values, kls = _compute_agent_outputs(
        agent=agent,
        mag_agent=mag_agent,
        observations=dataset.observation,
        actions=dataset.action,
        norm_factor=norm_factor,
    )

    # Normalize old log probs for comparison
    old_log_probs = dataset.log_prob / norm_factor

    # ========== Actor Loss ==========
    # PPO clipped policy loss
    ppo_loss, approx_kl, clip_frac = _compute_ppo_loss(
        log_probs, old_log_probs, normalized_advantages, valid_mask, clip_eps
    )

    # Entropy bonus (negative because we want to maximize entropy)
    entropy_loss = -_masked_mean(entropies, valid_mask)

    # Magnetic KL regularization (if mag_agent provided)
    mag_kl = _masked_mean(kls, valid_mask) if mag_agent is not None else jnp.float32(0.0)
    mag_loss = mag_kl

    # Total actor loss
    actor_loss = ppo_loss + ent_coef * entropy_loss + mag_coef * mag_loss

    # ========== Critic Loss ==========
    critic_loss = _compute_value_loss(
        values, dataset.value, dataset.target_value, valid_mask, clip_eps
    )

    # ========== Total Loss ==========
    total_loss = actor_loss + critic_loss

    # ========== Compute Auxiliary Losses ==========
    explained_var = _compute_explained_variance(values, dataset.target_value, valid_mask)

    aux_losses = {
        'actor_loss': actor_loss,
        'ppo_loss': ppo_loss,
        'critic_loss': critic_loss,
        'entropy': -entropy_loss,
        'mag_kl': mag_kl,
        'approx_kl': approx_kl,
        'clip_frac': clip_frac,
        'explained_var': explained_var
    }

    return total_loss, aux_losses
