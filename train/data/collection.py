"""
Trajectory collection from environments.
"""

from typing import Tuple
import numpy as np

import jax
import jax.numpy as jnp
import chex

import envs.mytypes as env_types
from agents import BaseAgent
from train.data.buffer import RolloutBuffer
import train.infrastructure.types as train_types


def collect_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_steps: int,
    buffer: RolloutBuffer
) -> Tuple[env_types.TimeStep, train_types.Transition, chex.Array, chex.Array]:
    """
    Collect trajectories from vectorized environments running on CPU.

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

    Returns:
        Tuple containing:
        - last_timestep: Final timestep (NumPy arrays)
        - transitions: Collected transitions as JAX arrays for training
                      Shape: (num_envs, num_steps, num_agents, ...)
        - next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        - next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                          Used to zero out bootstrap when episode naturally ended

    Note:
        - Environments auto-reset when all agents are done
        - NumPy ↔ JAX conversions happen at agent inference boundary
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

    buffer.reset()

    # Rollout loop (Python loop, can't JIT through NumPy env)
    for _ in range(num_steps):
        key, act_key = jax.random.split(key)

        # === Convert NumPy → JAX for agent inference ===
        obs_jax = jax.tree.map(jnp.asarray, last_timestep.observation)

        # Flatten env and agent dims for agent: (num_envs, num_agents, ...) → (batch, ...)
        batch_size = num_envs * num_agents
        def flatten_batch(x):
            return x.reshape(batch_size, *x.shape[2:])
        obs_flat = jax.tree.map(flatten_batch, obs_jax)

        # Agent forward pass (JAX, runs on GPU)
        actions_flat, log_probs_flat, values_flat = agent.get_action_and_value(
            obs_flat, act_key
        )

        # Unflatten back: (batch,) → (num_envs, num_agents, ...)
        actions_jax = actions_flat.reshape(num_envs, num_agents, *actions_flat.shape[1:])
        log_probs_jax = log_probs_flat.reshape(num_envs, num_agents)
        values_jax = values_flat.reshape(num_envs, num_agents)

        # === Convert JAX → NumPy for env step ===
        actions_np = np.asarray(actions_jax)
        values_np = np.asarray(values_jax)
        log_probs_np = np.asarray(log_probs_jax)

        # Step environment (NumPy, runs on CPU)
        new_timestep = env.step(actions_np)

        # Compute done flag as terminated OR truncated for episode boundary tracking
        done_np = last_timestep.terminated | last_timestep.truncated

        # Store in buffer (all NumPy)
        buffer.add(
            done=done_np,              # (num_envs, num_agents) - per-agent episode boundaries
            action=actions_np,               # (num_envs, num_agents, ...)
            value=values_np,                 # (num_envs, num_agents)
            reward=new_timestep.reward,      # (num_envs, num_agents)
            log_prob=log_probs_np,           # (num_envs, num_agents)
            observation=last_timestep.observation,  # (num_envs, num_agents, ...)
        )

        last_timestep = new_timestep

    # Compute bootstrap value for GAE (value of the next state after rollout)
    obs_jax = jax.tree.map(jnp.asarray, last_timestep.observation)
    batch_size = num_envs * num_agents
    def flatten_batch_bootstrap(x):
        return x.reshape(batch_size, *x.shape[2:])
    obs_flat = jax.tree.map(flatten_batch_bootstrap, obs_jax)

    # Get value estimate for bootstrap
    next_value_flat = agent.get_value(obs_flat)
    next_value = next_value_flat.reshape(num_envs, num_agents)
    # Only use terminated for bootstrap - if truncated, we still want to bootstrap
    next_terminated = jnp.asarray(last_timestep.terminated)  # (num_envs, num_agents)

    # Convert buffer to JAX arrays for GPU training
    transitions = buffer.to_jax()

    return last_timestep, transitions, next_value, next_terminated
