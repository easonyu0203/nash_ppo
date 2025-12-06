"""
Trajectory collection from environments.
"""

from typing import Tuple, Optional, Any
import numpy as np

import jax
import jax.numpy as jnp
import chex
from flax import nnx

import envs.mytypes as env_types
from agents import BaseAgent, StatefulAgent
from train.data.buffer import RolloutBuffer
import train.types as train_types


@nnx.jit(static_argnames=('num_envs', 'num_agents'))
def _agent_step_stateless(
    agent: BaseAgent,
    observation: env_types.Observation,
    key: chex.PRNGKey,
    num_envs: int,
    num_agents: int
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """JIT-compiled agent inference for stateless agents.

    Args:
        agent: Stateless agent
        observation: NumPy observation (will be converted to JAX)
        key: Random key for action sampling
        num_envs: Number of environments
        num_agents: Number of agents per environment

    Returns:
        Tuple of (actions, log_probs, values) with shape (num_envs, num_agents, ...)
    """
    batch_size = num_envs * num_agents

    # Convert NumPy → JAX
    obs_jax = jax.tree.map(jnp.asarray, observation)

    # Flatten env and agent dims: (num_envs, num_agents, ...) → (batch, ...)
    obs_flat = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        obs_jax
    )

    # Agent forward pass (runs on GPU)
    actions_flat, log_probs_flat, values_flat = agent.get_action_and_value(obs_flat, key)

    # Unflatten: (batch, ...) → (num_envs, num_agents, ...)
    actions = actions_flat.reshape(num_envs, num_agents, *actions_flat.shape[1:])
    log_probs = log_probs_flat.reshape(num_envs, num_agents)
    values = values_flat.reshape(num_envs, num_agents)

    return actions, log_probs, values


@nnx.jit(static_argnames=('num_envs', 'num_agents'), donate_argnames=['carries'])
def _agent_step_stateful(
    agent: StatefulAgent,
    observation: env_types.Observation,
    carries: Any,
    key: chex.PRNGKey,
    num_envs: int,
    num_agents: int
) -> Tuple[chex.Array, chex.Array, chex.Array, Any, Any]:
    """JIT-compiled agent inference for stateful agents.

    Args:
        agent: Stateful agent
        observation: NumPy observation (will be converted to JAX)
        carries: Batched hidden states (batch_size, *carry_shape)
        key: Random key for action sampling
        num_envs: Number of environments
        num_agents: Number of agents per environment

    Returns:
        Tuple of (actions, log_probs, values, carries, initial_carry_reshaped)
        - actions, log_probs, values: shape (num_envs, num_agents, ...)
        - carries: updated hidden states (batch_size, *carry_shape)
        - initial_carry_reshaped: initial carry reshaped to (num_envs, num_agents, *carry_shape)
    """
    batch_size = num_envs * num_agents

    # Store initial carry before step (for training)
    initial_carry = carries

    # Convert NumPy → JAX
    obs_jax = jax.tree.map(jnp.asarray, observation)

    # Flatten env and agent dims: (num_envs, num_agents, ...) → (batch, ...)
    obs_flat = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        obs_jax
    )

    # Agent forward pass with carries (runs on GPU)
    actions_flat, log_probs_flat, values_flat, new_carries = agent.get_action_and_value(
        obs_flat, carries, key
    )

    # Unflatten: (batch, ...) → (num_envs, num_agents, ...)
    actions = actions_flat.reshape(num_envs, num_agents, *actions_flat.shape[1:])
    log_probs = log_probs_flat.reshape(num_envs, num_agents)
    values = values_flat.reshape(num_envs, num_agents)

    # Reshape initial carry: (batch, *carry_shape) → (num_envs, num_agents, *carry_shape)
    initial_carry_reshaped = jax.tree.map(
        lambda x: x.reshape(num_envs, num_agents, *x.shape[1:]),
        initial_carry
    )

    return actions, log_probs, values, new_carries, initial_carry_reshaped


@nnx.jit(donate_argnames=['carries'])
def _reset_carries_at_done(carries: Any, done: chex.Array) -> Any:
    """JIT-compiled carry reset at episode boundaries.

    Args:
        carries: Hidden states (batch_size, *carry_shape)
        done: Done flags (batch_size,)

    Returns:
        Carries with done positions reset to zero
    """
    def reset_carry_at_done(carry_array):
        # carry_array shape: (batch, *carry_dims)
        fresh_carry = jnp.zeros_like(carry_array)
        # Use where: if done, use fresh_carry, else use current carry
        return jnp.where(
            done.reshape(carry_array.shape[0], *([1] * (carry_array.ndim - 1))),
            fresh_carry,
            carry_array
        )

    return jax.tree.map(reset_carry_at_done, carries)


@nnx.jit(static_argnames=('num_envs', 'num_agents'))
def _get_bootstrap_value_stateless(
    agent: BaseAgent,
    observation: env_types.Observation,
    key: chex.PRNGKey,
    num_envs: int,
    num_agents: int
) -> chex.Array:
    """JIT-compiled bootstrap value computation for stateless agents.

    Args:
        agent: Stateless agent
        observation: NumPy observation (will be converted to JAX)
        key: Random key for stochastic operations
        num_envs: Number of environments
        num_agents: Number of agents per environment

    Returns:
        Bootstrap values with shape (num_envs, num_agents)
    """
    batch_size = num_envs * num_agents

    # Convert NumPy → JAX and flatten
    obs_jax = jax.tree.map(jnp.asarray, observation)
    obs_flat = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        obs_jax
    )

    # Get value estimate
    value_flat = agent.get_value(obs_flat, key)
    return value_flat.reshape(num_envs, num_agents)


@nnx.jit(static_argnames=('num_envs', 'num_agents'))
def _get_bootstrap_value_stateful(
    agent: StatefulAgent,
    observation: env_types.Observation,
    carries: Any,
    key: chex.PRNGKey,
    num_envs: int,
    num_agents: int
) -> chex.Array:
    """JIT-compiled bootstrap value computation for stateful agents.

    Args:
        agent: Stateful agent
        observation: NumPy observation (will be converted to JAX)
        carries: Batched hidden states (batch_size, *carry_shape)
        key: Random key for stochastic operations
        num_envs: Number of environments
        num_agents: Number of agents per environment

    Returns:
        Bootstrap values with shape (num_envs, num_agents)
    """
    batch_size = num_envs * num_agents

    # Convert NumPy → JAX and flatten
    obs_jax = jax.tree.map(jnp.asarray, observation)
    obs_flat = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        obs_jax
    )

    # Get value estimate with carries
    value_flat, _ = agent.get_value(obs_flat, carries, key)
    return value_flat.reshape(num_envs, num_agents)


def collect_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_steps: int,
    num_envs: int,
    num_agents: int,
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
        num_envs: Number of parallel environments
        num_agents: Number of agents per environment
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

        # === JIT-compiled agent inference (runs on GPU) ===
        if is_stateful:
            # Stateful agent: use JIT-compiled helper
            actions_jax, log_probs_jax, values_jax, carries, initial_carry_jax = _agent_step_stateful(
                agent, last_timestep.observation, carries, act_key, num_envs, num_agents
            )
        else:
            # Stateless agent: use JIT-compiled helper
            actions_jax, log_probs_jax, values_jax = _agent_step_stateless(
                agent, last_timestep.observation, act_key, num_envs, num_agents
            )
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

        # Reset carries at episode boundaries for stateful agents (JIT-compiled)
        if is_stateful:
            done_flat_jax = jnp.asarray(done_np.reshape(batch_size))  # (batch,)
            carries = _reset_carries_at_done(carries, done_flat_jax)

        last_timestep = new_timestep

    # Compute bootstrap value for GAE (value of the next state after rollout) - JIT-compiled
    key, bootstrap_key = jax.random.split(key)
    if is_stateful:
        next_value = _get_bootstrap_value_stateful(
            agent, last_timestep.observation, carries, bootstrap_key, num_envs, num_agents
        )
    else:
        next_value = _get_bootstrap_value_stateless(
            agent, last_timestep.observation, bootstrap_key, num_envs, num_agents
        )

    # Only use terminated for bootstrap - if truncated, we still want to bootstrap
    next_terminated = jnp.asarray(last_timestep.terminated)  # (num_envs, num_agents)

    # Convert buffer to JAX arrays for GPU training
    transitions = buffer.to_jax()

    # Return carries for stateful agents to maintain continuity across rollouts
    return last_timestep, transitions, next_value, next_terminated, carries
