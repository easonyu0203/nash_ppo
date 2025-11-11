"""
Transition processing and advantage computation.
"""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
import chex

import train.types as train_types
from train.algorithms.value_norm import ValueNorm


def rearrange_transitions(transitions: train_types.Transition) -> train_types.Transition:
    """
    Rearrange transitions from (num_envs, num_steps, num_agents) to (num_envs, num_agents, num_steps).

    This makes the data natural for:
    - GAE calculation (batch of trajectories over time)
    - Flattening (num_envs * num_agents becomes batch dimension)
    """
    def swap_axes(x: chex.Array) -> chex.Array:
        """Swap step and agent dimensions: (num_envs, num_steps, num_agents, ...) -> (num_envs, num_agents, num_steps, ...)"""
        return jnp.swapaxes(x, 1, 2)

    return train_types.Transition(
        done=swap_axes(transitions.done),  # (num_envs, num_steps, num_agents) -> (num_envs, num_agents, num_steps)
        action=swap_axes(transitions.action),
        value=swap_axes(transitions.value),
        reward=swap_axes(transitions.reward),
        log_prob=swap_axes(transitions.log_prob),
        observation=jax.tree.map(swap_axes, transitions.observation),
        initial_carry=jax.tree.map(swap_axes, transitions.initial_carry) if transitions.initial_carry is not None else None,
    )


def create_dataset(
    transitions: train_types.Transition,
    advantages: chex.Array,
    target_values: chex.Array,
    batch_size: int,
    bptt_length: Optional[int] = None
) -> train_types.Dataset:
    """
    Create dataset from rearranged transitions.

    For stateless agents (bptt_length=None):
        Flattens to shape (batch_size, ...) where batch_size = num_envs * num_agents * num_steps

    For stateful agents (bptt_length specified):
        Reshapes to (batch_size, bptt_length, ...) where batch_size = num_envs * num_agents * num_steps // bptt_length
        Each sample is a sequence of bptt_length timesteps for BPTT training

    Args:
        transitions: Rearranged transitions with shape (num_envs, num_agents, num_steps, ...)
        advantages: Shape (num_envs, num_agents, num_steps)
        target_values: Shape (num_envs, num_agents, num_steps)
        batch_size: num_envs * num_agents * num_steps
        bptt_length: If specified, reshape into BPTT segments of this length

    Returns:
        Dataset with fields shaped appropriately for stateless or stateful training
    """
    if bptt_length is None:
        # Stateless: flatten everything to (batch_size, ...)
        def flatten_to_batch(x: chex.Array) -> chex.Array:
            """Flatten (num_envs, num_agents, num_steps, ...) -> (batch_size, ...)"""
            return x.reshape(batch_size, *x.shape[3:])

        return train_types.Dataset(
            action=flatten_to_batch(transitions.action),
            value=transitions.value.reshape(batch_size),
            log_prob=transitions.log_prob.reshape(batch_size),
            observation=jax.tree.map(flatten_to_batch, transitions.observation),
            advantage=advantages.reshape(batch_size),
            target_value=target_values.reshape(batch_size),
            done=transitions.done.reshape(batch_size),
            initial_carry=None,
        )
    else:
        # Stateful: reshape to BPTT segments (batch_size // bptt_length, bptt_length, ...)
        num_envs, num_agents, num_steps = transitions.reward.shape
        num_sequences = batch_size // bptt_length

        def reshape_to_sequences(x: chex.Array) -> chex.Array:
            """Reshape (num_envs, num_agents, num_steps, ...) -> (num_sequences, bptt_length, ...)"""
            # First flatten to (num_envs * num_agents, num_steps, ...)
            flat_shape = (num_envs * num_agents, num_steps, *x.shape[3:])
            x_flat = x.reshape(flat_shape)
            # Then reshape to (num_envs * num_agents * num_steps // bptt_length, bptt_length, ...)
            seq_shape = (num_sequences, bptt_length, *x.shape[3:])
            return x_flat.reshape(seq_shape)

        def reshape_scalar_to_sequences(x: chex.Array) -> chex.Array:
            """Reshape scalar arrays (num_envs, num_agents, num_steps) -> (num_sequences, bptt_length)"""
            # Flatten to (num_envs * num_agents, num_steps)
            x_flat = x.reshape(num_envs * num_agents, num_steps)
            # Reshape to (num_sequences, bptt_length)
            return x_flat.reshape(num_sequences, bptt_length)

        # For carries, we only need the initial carry at the start of each BPTT segment
        # Shape: (num_envs, num_agents, num_steps, *carry_shape) -> sample every bptt_length
        initial_carry_sequences = None
        def extract_segment_starts(carry_array: chex.Array) -> chex.Array:
            """Extract carries at segment boundaries.
            Input: (num_envs, num_agents, num_steps, *carry_dims)
            Output: (num_sequences, *carry_dims)
            """
            # Flatten to (num_envs * num_agents, num_steps, *carry_dims)
            flat_shape = (num_envs * num_agents, num_steps, *carry_array.shape[3:])
            carry_flat = carry_array.reshape(flat_shape)
            # Reshape to (num_sequences, bptt_length, *carry_dims)
            carry_seqs = carry_flat.reshape(num_sequences, bptt_length, *carry_array.shape[3:])
            # Take only the first timestep of each sequence (index 0)
            return carry_seqs[:, 0, ...]  # (num_sequences, *carry_dims)

        initial_carry_sequences = jax.tree.map(extract_segment_starts, transitions.initial_carry)

        return train_types.Dataset(
            action=reshape_to_sequences(transitions.action),
            value=reshape_scalar_to_sequences(transitions.value),
            log_prob=reshape_scalar_to_sequences(transitions.log_prob),
            observation=jax.tree.map(reshape_to_sequences, transitions.observation),
            advantage=reshape_scalar_to_sequences(advantages),
            target_value=reshape_scalar_to_sequences(target_values),
            done=reshape_scalar_to_sequences(transitions.done),
            initial_carry=initial_carry_sequences,
        )


@nnx.jit(static_argnames=('bptt_length',))
def process_transitions(
    transitions: train_types.Transition,
    metrics: nnx.MultiMetric,
    next_value: chex.Array,
    next_terminated: chex.Array,
    gamma: float,
    gae_gamma: float,
    value_normalizer: Optional[ValueNorm] = None,
    bptt_length: Optional[int] = None,
) -> Tuple[nnx.MultiMetric, train_types.Dataset]:
    """
    Process transitions and calculate advantages using GAE.

    Args:
        transitions: Collected transitions with shape (num_envs, num_steps, num_agents, ...)
        metrics: Metric tracker for logging
        next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                        Used to zero out bootstrap when episode naturally ended (not truncated)
        gamma: Discount factor
        gae_gamma: GAE lambda parameter
        value_normalizer: Optional value normalizer for denormalizing value predictions
        bptt_length: If specified, creates sequences for BPTT training (stateful agents)

    Returns:
        Tuple of (updated metrics, training dataset)
    """
    # Step 2: Rearrange to (num_envs, num_agents, num_steps) for processing
    transitions = rearrange_transitions(transitions)
    num_envs, num_agents, num_steps = transitions.reward.shape

    # Step 3: Calculate GAE - works on (num_envs, num_agents, num_steps)
    advantages, target_values = calculate_gae(
        transitions, next_value, next_terminated, gamma, gae_gamma, value_normalizer
    )

    # Step 4: Create dataset (flatten for stateless, reshape to sequences for stateful)
    batch_size = num_envs * num_agents * num_steps

    dataset = create_dataset(
        transitions, advantages, target_values, batch_size, bptt_length
    )

    # Log metrics - agent 0 reward only
    ag0_reward = transitions.reward[:, 0, :]  # (num_envs, num_steps)
    ag0_dones = transitions.done[:, 0, :]  # (num_envs, num_steps)
    metrics.update(
        inverse_eps_len=ag0_dones.reshape(num_envs * num_steps),
        reward=ag0_reward.reshape(num_envs * num_steps)
    )

    return metrics, dataset


@nnx.jit
def calculate_gae(
    transitions: train_types.Transition,
    next_value: chex.Array,
    next_terminated: chex.Array,
    gamma: float,
    gae_gamma: float,
    value_normalizer: Optional[ValueNorm] = None,
) -> Tuple[chex.Array, chex.Array]:
    """
    Calculate Generalized Advantage Estimation (GAE) for multi-agent trajectories.

    Strategy: Work directly on (num_envs, num_agents, num_steps) shape - flatten once and vmap.

    Args:
        transitions: trajectory data with shape (num_envs, num_agents, num_steps, ...)
                    - reward: (num_envs, num_agents, num_steps)
                    - value: (num_envs, num_agents, num_steps)
                    - done: (num_envs, num_agents, num_steps) - per-agent episode boundaries
        next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                        When True, episode naturally ended → don't bootstrap (use 0)
                        When False, episode was truncated → DO bootstrap (use next_value)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        gae_gamma: GAE lambda parameter (0 < gae_gamma <= 1)
        value_normalizer: Optional value normalizer for denormalizing value predictions

    Returns:
        Tuple of (advantages, target_values) with shape (num_envs, num_agents, num_steps)
    """

    def calculate_single_trajectory_gae(
        trajectory_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Calculate GAE for a single trajectory (one env, one agent).

        Args:
            trajectory_data: Tuple of (rewards, values, dones, next_value, next_terminated)
                           - rewards, values, dones: shape (num_steps,)
                           - next_value: scalar
                           - next_terminated: scalar - True if naturally ended, False if truncated

        Returns:
            Tuple of (advantages, target_values), each shape (num_steps,)
        """
        rewards, values, dones, next_value, next_terminated = trajectory_data

        def gae_step(
            carry: Tuple[chex.Array, chex.Array],
            timestep_data: Tuple[chex.Array, chex.Array, chex.Array]
        ) -> Tuple[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]:
            """Single GAE step in reverse scan over time."""
            next_gae, next_value = carry
            reward, value, done = timestep_data

            # Compute TD error and GAE
            td_error = reward + gamma * next_value - value
            gae = td_error + gamma * gae_gamma * next_gae

            advantage = gae
            target_value = advantage + value

            # Reset carry at episode boundaries (prevent leakage to previous episode)
            gae_out = jnp.where(done, 0.0, gae)
            value_out = jnp.where(done, 0.0, value)

            return (gae_out, value_out), (advantage, target_value)

        # - If terminated (natural end): use 0 (no future value)
        # - If truncated (time limit): use next_value (episode continues, we just stopped collecting)
        bootstrap_value = jnp.where(next_terminated, 0.0, next_value)
        init_carry = (jnp.float32(0.0), bootstrap_value)

        _, (advantages, target_values) = jax.lax.scan(
            gae_step,
            init_carry,
            (rewards, values, dones),
            reverse=True
        )

        return advantages, target_values

    # Extract data - already in shape (num_envs, num_agents, num_steps)
    rewards = transitions.reward
    values = transitions.value
    dones = transitions.done  # Already (num_envs, num_agents, num_steps)

    # Denormalize values if using value normalization
    # Values from the network are in normalized space, need to convert to original scale for GAE
    if value_normalizer is not None:
        # Flatten for denormalization
        num_envs, num_agents, num_steps = values.shape
        values_flat = values.reshape(-1)
        next_value_flat = next_value.reshape(-1)

        # Denormalize
        values_denorm_flat = value_normalizer.denormalize(values_flat)
        next_value_denorm_flat = value_normalizer.denormalize(next_value_flat)

        # Reshape back
        values = values_denorm_flat.reshape(num_envs, num_agents, num_steps)
        next_value = next_value_denorm_flat.reshape(num_envs, num_agents)

    # Infer shapes from arrays
    num_envs, num_agents, num_steps = rewards.shape

    # Flatten to batch: (num_envs, num_agents, ...) → (num_envs * num_agents, ...)
    batch_size = num_envs * num_agents
    rewards_flat = rewards.reshape(batch_size, num_steps)
    values_flat = values.reshape(batch_size, num_steps)
    dones_flat = dones.reshape(batch_size, num_steps)
    next_value_flat = next_value.reshape(batch_size)
    next_terminated_flat = next_terminated.reshape(batch_size)

    # Vmap over batch dimension
    batched_gae = jax.vmap(calculate_single_trajectory_gae, in_axes=0, out_axes=0)
    advantages_flat, target_values_flat = batched_gae((
        rewards_flat, values_flat, dones_flat, next_value_flat, next_terminated_flat
    ))

    # Unflatten: (num_envs * num_agents, num_steps) → (num_envs, num_agents, num_steps)
    advantages = advantages_flat.reshape(num_envs, num_agents, num_steps)
    target_values = target_values_flat.reshape(num_envs, num_agents, num_steps)

    return advantages, target_values
