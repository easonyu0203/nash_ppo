"""PPO training for stateful agents (BPTT sequences)."""

import jax
import jax.numpy as jnp
import chex

import train.infrastructure.types as train_types
from agents import StatefulAgent
from train.algorithms.ppo_utils import get_action_norm_factor

# Constants
_EPSILON = 1e-8


def _swap_leading_axes(pytree):
    """Swap first two axes of all arrays in pytree. Works for both arrays and pytrees."""
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), pytree)


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


def _compute_bptt_outputs(
    agent: StatefulAgent,
    mag_agent: StatefulAgent | None,
    observations: chex.ArrayTree,
    actions: jnp.ndarray,
    dones: jnp.ndarray,
    initial_carries: chex.ArrayTree,
    bptt_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run BPTT forward pass through sequences to get agent outputs.

    Scans over time dimension with batched carries (one carry per sequence in batch).
    The agent processes all sequences in parallel at each timestep, with each sequence
    maintaining its own hidden state.

    IMPORTANT: Resets hidden states at episode boundaries (when done=True) to prevent
    hidden state leakage across episodes.

    Args:
        agent: Agent to evaluate
        mag_agent: Optional magnetic agent for KL computation
        observations: Shape (batch_size, bptt_length, *obs_shape) - can be pytree or array
        actions: Shape (batch_size, bptt_length, action_dim)
        dones: Shape (batch_size, bptt_length) - episode boundaries for resetting carries
        initial_carries: Shape (batch_size, *carry_shape) - batched carries, one per sequence
        bptt_length: Length of BPTT sequences

    Returns:
        Tuple of (log_probs, entropies, values, kls) with shape (batch_size, bptt_length)
    """
    def process_timestep(carries, timestep_data):
        """Process one timestep for all sequences in batch.

        The agent handles batched observations and carries directly - no vmap needed.
        Each sequence in the batch has its own carry that gets updated independently.

        Args:
            carries: Batched carries with shape (batch_size, *carry_shape)
            timestep_data: Tuple of (obs, actions, dones) all with shape (batch_size, ...)

        Returns:
            Tuple of (new_carries, outputs) where:
            - new_carries has shape (batch_size, *carry_shape)
            - outputs are (log_probs, entropies, values, kls) each with shape (batch_size,)
        """
        obs_t, action_t, done_t = timestep_data  # (batch_size, *obs_shape), (batch_size, action_dim), (batch_size,)

        # Agent handles batched observations and carries directly
        dist, values, new_carries = agent.get_distribution_and_value(obs_t, carries)

        # Compute policy outputs
        log_probs = dist.log_prob(action_t)
        entropies = dist.entropy()

        # Compute KL with magnetic agent if available
        kls = jnp.zeros_like(log_probs)
        if mag_agent is not None:
            mag_dist, _, _ = mag_agent.get_distribution_and_value(obs_t, carries)
            kls = dist.kl_divergence(mag_dist)

        # Reset carries at episode boundaries to prevent hidden state leakage
        def reset_carry_at_done(carry_array):
            """Reset carry to zero when done=True."""
            fresh_carry = jnp.zeros_like(carry_array)
            # Reshape done to broadcast: (batch_size,) -> (batch_size, 1, 1, ...)
            done_expanded = done_t.reshape(done_t.shape[0], *([1] * (carry_array.ndim - 1)))
            return jnp.where(done_expanded, fresh_carry, carry_array)

        new_carries = jax.tree.map(reset_carry_at_done, new_carries)

        return new_carries, (log_probs, entropies, values, kls)

    # Transpose to (bptt_length, batch_size, ...) for scanning over time
    obs_time_major = _swap_leading_axes(observations)
    actions_time_major = _swap_leading_axes(actions)
    dones_time_major = _swap_leading_axes(dones)

    # Scan over time dimension, processing all sequences at each timestep
    _, outputs = jax.lax.scan(
        process_timestep,
        initial_carries,
        (obs_time_major, actions_time_major, dones_time_major),
        length=bptt_length
    )

    # Transpose outputs back to (batch_size, bptt_length)
    log_probs, entropies, values, kls = outputs
    log_probs = jnp.swapaxes(log_probs, 0, 1)
    entropies = jnp.swapaxes(entropies, 0, 1)
    values = jnp.swapaxes(values, 0, 1)
    kls = jnp.swapaxes(kls, 0, 1)

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


def calculate_loss_stateful(
    agent: StatefulAgent,
    mag_agent: StatefulAgent,
    dataset: train_types.Dataset,
    ent_coef: float,
    mag_coef: float,
    clip_eps: float,
    normalize_logprob: bool,
) -> tuple[chex.Numeric, dict[str, chex.Numeric]]:
    """Calculate PPO loss for stateful agents using BPTT.

    Dataset shape: (batch_size, bptt_length, ...)
    Each sample is a sequence of bptt_length timesteps for BPTT training.

    Args:
        agent: Stateful agent to train
        mag_agent: Magnetic agent for regularization (can be None)
        dataset: Training dataset with sequences
        ent_coef: Entropy coefficient
        mag_coef: Magnetic loss coefficient
        clip_eps: PPO clipping epsilon
        normalize_logprob: Whether to normalize log probs by action dimensions

    Returns:
        Tuple of (total_loss, aux_losses) where aux_losses is a dict containing:
        - actor_loss, ppo_loss, critic_loss, entropy, mag_kl, approx_kl, clip_frac, explained_var
    """
    _, bptt_length = dataset.advantage.shape
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

    # ========== BPTT Forward Pass ==========
    # Run sequences through agent to get current policy outputs
    log_probs, entropies, values, kls = _compute_bptt_outputs(
        agent=agent,
        mag_agent=mag_agent,
        observations=dataset.observation,
        actions=dataset.action,
        dones=dataset.done,
        initial_carries=dataset.initial_carry,
        bptt_length=bptt_length,
    )

    # ========== Prepare Data for Loss Computation ==========
    # Flatten sequences: (batch_size, bptt_length) -> (batch_size * bptt_length)
    flatten = lambda x: x.reshape(-1)

    log_probs_flat = flatten(log_probs) / norm_factor
    old_log_probs_flat = flatten(dataset.log_prob) / norm_factor
    entropies_flat = flatten(entropies) / norm_factor
    values_flat = flatten(values)
    old_values_flat = flatten(dataset.value)
    advantages_flat = flatten(normalized_advantages)
    target_values_flat = flatten(dataset.target_value)
    kls_flat = flatten(kls) / norm_factor
    valid_flat = flatten(valid_mask)

    # ========== Actor Loss ==========
    # PPO clipped policy loss
    ppo_loss, approx_kl, clip_frac = _compute_ppo_loss(
        log_probs_flat, old_log_probs_flat, advantages_flat, valid_flat, clip_eps
    )

    # Entropy bonus (negative because we want to maximize entropy)
    entropy_loss = -_masked_mean(entropies_flat, valid_flat)

    # Magnetic KL regularization (if mag_agent provided)
    mag_kl = _masked_mean(kls_flat, valid_flat) if mag_agent is not None else jnp.float32(0.0)
    mag_loss = mag_kl

    # Total actor loss
    actor_loss = ppo_loss + ent_coef * entropy_loss + mag_coef * mag_loss

    # ========== Critic Loss ==========
    critic_loss = _compute_value_loss(
        values_flat, old_values_flat, target_values_flat, valid_flat, clip_eps
    )

    # ========== Total Loss ==========
    total_loss = actor_loss + critic_loss

    # ========== Compute Auxiliary Losses ==========
    explained_var = _compute_explained_variance(values_flat, target_values_flat, valid_flat)

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
