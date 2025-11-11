"""
Rollout buffer for CPU-side trajectory storage.
"""

from typing import Optional, Any
import numpy as np
import jax
import jax.numpy as jnp

import train.types as train_types


class RolloutBuffer:
    """
    CPU-side buffer for storing rollout data before transferring to GPU.
    Stores data as NumPy arrays for efficient CPU environment interaction.
    Supports both array and Dict observation spaces.
    """
    def __init__(self, num_envs: int, num_steps: int, num_agents: int, obs_shape, action_shape: tuple):
        """
        Args:
            num_envs: Number of parallel environments
            num_steps: Number of steps to collect
            num_agents: Number of agents per environment
            obs_shape: Shape of observations (per agent) - can be tuple for array obs or dict for Dict obs
            action_shape: Shape of actions (per agent)
                        - Discrete: () - scalar
                        - MultiDiscrete: (n,) - vector
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.step = 0

        # Preallocate buffers (NumPy arrays on CPU)
        self.dones = np.zeros((num_envs, num_steps, num_agents), dtype=np.bool_)
        self.actions = np.zeros((num_envs, num_steps, num_agents, *action_shape), dtype=np.int32)
        self.values = np.zeros((num_envs, num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_envs, num_steps, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((num_envs, num_steps, num_agents), dtype=np.float32)

        # Support Dict observations
        if isinstance(obs_shape, dict):
            self.observations = {
                key: np.zeros((num_envs, num_steps, num_agents, *shape), dtype=np.float32)
                for key, shape in obs_shape.items()
            }
        else:
            self.observations = np.zeros((num_envs, num_steps, num_agents, *obs_shape), dtype=np.float32)

        # Carry storage (None for stateless agents)
        self.initial_carries: Optional[Any] = None

    def enable_carry_storage(self, carry_spec):
        """Enable storage of carries for recurrent agents.

        Must be called before collection starts if using stateful agents.

        Args:
            carry_spec: Pytree of ShapeDtypeStruct from agent.get_carry_spec(batch_size)
                       where batch_size = num_envs * num_agents
                       Shape: (batch_size, *carry_dims)
        """
        # Allocate buffer for carries: (num_envs, num_steps, num_agents, *carry_shape)
        # carry_spec has shape (batch_size, *carry_dims) where batch_size = num_envs * num_agents
        def allocate_carry_array(spec: jax.ShapeDtypeStruct):
            # spec.shape is (batch_size, *carry_dims) where batch_size = num_envs * num_agents
            # We want: (num_envs, num_steps, num_agents, *carry_dims)
            carry_dims = spec.shape[1:]  # Extract everything after batch dim
            return np.zeros((self.num_envs, self.num_steps, self.num_agents, *carry_dims), dtype=spec.dtype)

        self.initial_carries = jax.tree.map(allocate_carry_array, carry_spec)

    def reset(self):
        """Reset buffer for new collection"""
        self.step = 0

    def add(self, done, action, value, reward, log_prob, observation, initial_carry=None):
        """Add a timestep of data to buffer

        Args:
            done: Per-agent episode boundary flags, shape (num_envs, num_agents)
            action: Actions taken, shape (num_envs, num_agents, ...)
            value: Value estimates, shape (num_envs, num_agents)
            reward: Rewards received, shape (num_envs, num_agents)
            log_prob: Log probabilities, shape (num_envs, num_agents)
            observation: Observations, shape (num_envs, num_agents, ...) or dict of such
            initial_carry: Optional hidden state at start of timestep
                          Shape: (num_envs, num_agents, *carry_shape)
        """
        self.dones[:, self.step, :] = done
        self.actions[:, self.step] = action
        self.values[:, self.step] = value
        self.rewards[:, self.step] = reward
        self.log_probs[:, self.step] = log_prob

        # Handle Dict observations
        if isinstance(self.observations, dict):
            for key in self.observations.keys():
                self.observations[key][:, self.step] = observation[key]
        else:
            self.observations[:, self.step] = observation

        # Store carries if provided (already in correct shape: (num_envs, num_agents, *carry_shape))
        if initial_carry is not None:
            assert self.initial_carries is not None, (
                "Received initial_carry but buffer carry storage not enabled. "
                "Call enable_carry_storage() before collecting trajectories with stateful agents."
            )

            def store_carry(buffer_array, carry_array_np):
                # carry_array_np shape: (num_envs, num_agents, *carry_dims) - NumPy array
                buffer_array[:, self.step, :] = carry_array_np
                return buffer_array

            self.initial_carries = jax.tree.map(
                store_carry,
                self.initial_carries,
                initial_carry
            )

        self.step += 1

    def to_jax(self) -> train_types.Transition:
        """Convert buffer to JAX arrays for GPU training"""
        return train_types.Transition(
            done=jnp.asarray(self.dones),
            action=jnp.asarray(self.actions),
            value=jnp.asarray(self.values),
            reward=jnp.asarray(self.rewards),
            log_prob=jnp.asarray(self.log_probs),
            observation=jax.tree.map(jnp.asarray, self.observations),
            initial_carry=jax.tree.map(jnp.asarray, self.initial_carries) if self.initial_carries is not None else None,
        )
