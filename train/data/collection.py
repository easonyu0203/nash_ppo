"""
Trajectory collection from environments.
"""

from typing import Tuple, Optional, Any
import numpy as np

import jax
import jax.numpy as jnp
import chex

import envs.mytypes as env_types
from agents import BaseAgent, StatefulAgent
from train.data.buffer import RolloutBuffer
import train.infrastructure.types as train_types


def collect_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_steps: int,
    buffer: RolloutBuffer,
    carries: Optional[Any] = None
) -> Tuple[env_types.TimeStep, train_types.Transition, chex.Array, chex.Array, Optional[Any]]:
    """
    Collect trajectories from vectorized environments running on CPU.

    Supports both stateless and stateful (recurrent) agents.
    For stateful agents, manages hidden states and resets them at episode boundaries.
    Maintains temporal continuity by accepting and returning carries across rollouts.

    Data flow:
    1. Env step on CPU (NumPy) → store in buffer
    2. Agent inference on GPU (JAX)
    3. Convert buffer to JAX at end for training

    Args:
        env: Vectorized environment (e.g., DummyVecEnv)
             Returns data with shape (num_envs, num_agents, ...)
        agent: Shared agent model (JAX-based, runs on GPU)
        last_timestep: Last timestep from previous rollout (NumPy arrays)
                      Fields have shape (num_envs, num_agents, ...)
        key: JAX random key for action sampling
        num_steps: Number of steps to collect
        buffer: Preallocated RolloutBuffer for CPU-side storage
        carries: Optional hidden states from previous rollout (for stateful agents)
                Shape: (batch_size, *carry_shape) where batch_size = num_envs * num_agents
                Batched carries - one hidden state per (env, agent) pair
                If None and agent is stateful, will initialize fresh carries

    Returns:
        Tuple containing:
        - last_timestep: Final timestep (NumPy arrays)
        - transitions: Collected transitions as JAX arrays for training
                      Shape: (num_envs, num_steps, num_agents, ...)
        - next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        - next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                          Used to zero out bootstrap when episode naturally ended
        - carries: Final hidden states after rollout (for stateful agents), None otherwise
                  Shape: (batch_size, *carry_shape) where batch_size = num_envs * num_agents

    Note:
        - Environments auto-reset when all agents are done
        - NumPy ↔ JAX conversions happen at agent inference boundary
        - For stateful agents, carries are managed and reset at episode boundaries
        - Carries maintain temporal continuity across rollouts
    """
    # Get shapes from last_timestep
    # Handle Dict observations - get shape from first key
    if isinstance(last_timestep.observation, dict):
        first_key = next(iter(last_timestep.observation.keys()))
        num_envs = last_timestep.observation[first_key].shape[0]
        num_agents = last_timestep.observation[first_key].shape[1]
    else:
        num_envs = last_timestep.observation.shape[0]
        num_agents = last_timestep.observation.shape[1]

    batch_size = num_envs * num_agents
    buffer.reset()

    # Check if agent is stateful (recurrent)
    is_stateful = isinstance(agent, StatefulAgent)

    # Initialize or use existing carries for stateful agents
    if is_stateful:
        assert carries is not None, (
            "Stateful agent requires carries to be initialized before collection. "
            "Ensure LearnerState.carries is initialized in create_learner_state()."
        )

    # Rollout loop (Python loop, can't JIT through NumPy env)
    for _ in range(num_steps):
        key, act_key = jax.random.split(key)

        # === Convert NumPy → JAX for agent inference ===
        obs_jax = jax.tree.map(jnp.asarray, last_timestep.observation)

        # Flatten env and agent dims for agent: (num_envs, num_agents, ...) → (batch, ...)
        def flatten_batch(x):
            return x.reshape(batch_size, *x.shape[2:])
        obs_flat = jax.tree.map(flatten_batch, obs_jax)

        # Agent forward pass (JAX, runs on GPU)
        if is_stateful:
            # Store initial carry before step (for training)
            initial_carry = carries

            # Stateful agent: pass carries
            actions_flat, log_probs_flat, values_flat, carries = agent.get_action_and_value(
                obs_flat, carries, act_key
            )
        else:
            # Stateless agent: no carries
            actions_flat, log_probs_flat, values_flat = agent.get_action_and_value(
                obs_flat, act_key
            )
            initial_carry = None

        # Unflatten back: (batch,) → (num_envs, num_agents, ...)
        actions_jax = actions_flat.reshape(num_envs, num_agents, *actions_flat.shape[1:])
        log_probs_jax = log_probs_flat.reshape(num_envs, num_agents)
        values_jax = values_flat.reshape(num_envs, num_agents)

        # Reshape carry: (batch, *carry_shape) → (num_envs, num_agents, *carry_shape)
        if initial_carry is not None:
            initial_carry_jax = jax.tree.map(
                lambda x: x.reshape(num_envs, num_agents, *x.shape[1:]),
                initial_carry
            )
        else:
            initial_carry_jax = None

        # === Convert JAX → NumPy for env step ===
        actions_np = np.asarray(actions_jax)
        values_np = np.asarray(values_jax)
        log_probs_np = np.asarray(log_probs_jax)
        initial_carry_np = jax.tree.map(np.asarray, initial_carry_jax) if initial_carry_jax is not None else None

        # Step environment (NumPy, runs on CPU)
        new_timestep = env.step(actions_np)

        # Compute done flag as terminated OR truncated for episode boundary tracking
        done_np = last_timestep.terminated | last_timestep.truncated

        # Store in buffer (all NumPy)
        buffer.add(
            done=done_np,                                   # (num_envs, num_agents)
            action=actions_np,                              # (num_envs, num_agents, ...)
            value=values_np,                                # (num_envs, num_agents)
            reward=new_timestep.reward,                     # (num_envs, num_agents)
            log_prob=log_probs_np,                          # (num_envs, num_agents)
            observation=last_timestep.observation,          # (num_envs, num_agents, ...)
            initial_carry=initial_carry_np                  # (num_envs, num_agents, *carry_shape) or None
        )

        # Reset carries at episode boundaries for stateful agents
        if is_stateful:
            # Check which (env, agent) pairs are done
            done_flat = done_np.reshape(batch_size)  # (batch,)

            # Reset carries for done trajectories
            # We need to replace carries at done positions with fresh initialized carries
            def reset_carry_at_done(carry_array):
                # carry_array shape: (batch, *carry_dims)
                # Create fresh carry for this position
                fresh_carry = jnp.zeros_like(carry_array)
                # Use where: if done, use fresh_carry, else use current carry
                return jnp.where(
                    done_flat.reshape(batch_size, *([1] * (carry_array.ndim - 1))),
                    fresh_carry,
                    carry_array
                )

            carries = jax.tree.map(reset_carry_at_done, carries)

        last_timestep = new_timestep

    # Compute bootstrap value for GAE (value of the next state after rollout)
    obs_jax = jax.tree.map(jnp.asarray, last_timestep.observation)
    def flatten_batch_bootstrap(x):
        return x.reshape(batch_size, *x.shape[2:])
    obs_flat = jax.tree.map(flatten_batch_bootstrap, obs_jax)

    # Get value estimate for bootstrap
    if is_stateful:
        next_value_flat, _ = agent.get_value(obs_flat, carries)
    else:
        next_value_flat = agent.get_value(obs_flat)

    next_value = next_value_flat.reshape(num_envs, num_agents)
    # Only use terminated for bootstrap - if truncated, we still want to bootstrap
    next_terminated = jnp.asarray(last_timestep.terminated)  # (num_envs, num_agents)

    # Convert buffer to JAX arrays for GPU training
    transitions = buffer.to_jax()

    # Return carries for stateful agents to maintain continuity across rollouts
    return last_timestep, transitions, next_value, next_terminated, carries
