from pathlib import Path
from typing import Tuple, Any, Optional
from flax import nnx
import chex
import abc
import distrax
import orbax.checkpoint as ocp
import json
import jax

import envs.mytypes as env_types
from agents.base.checkpointable_mixin import CheckpointableMixin


class BaseAgent(nnx.Module, CheckpointableMixin):
    """Base class for all RL agents with checkpointing support.

    This is a CONCRETE class (not abstract) that provides common functionality
    for all agents, primarily checkpointing infrastructure. It does NOT define
    any abstract methods for agent behavior.

    Agent behavior is defined by subclasses:
    - StatelessAgent: For agents without hidden state (MLP, CNN)
    - StatefulAgent: For agents with hidden state (LSTM, GRU)

    This class combines:
    - nnx.Module: For neural network parameters and JAX integration
    - CheckpointableMixin: For automatic configuration tracking

    Provides:
    - save_checkpoint: Save agent parameters and config to disk
    - load_checkpoint: Load agent from disk with correct class instantiation

    Note: This class is intentionally minimal to avoid duplication. All agent-specific
          behavior (forward passes, etc.) is defined in subclasses.
    """
    pass

    def save_checkpoint(self, checkpoint_dir: str, step: int):
        """Save agent checkpoint to disk using Orbax.

        Saves three components:
        1. agent_state: Neural network parameters (Orbax format)
        2. metadata.json: Agent class name and configuration

        Best practices followed:
        - Device-agnostic: Arrays are moved to CPU before saving to ensure
          compatibility across different devices (CPU/CUDA/TPU)
        - State only: Only saves trainable parameters, not graph structure

        Args:
            checkpoint_dir: Directory to save checkpoint
            step: step number for checkpoint naming
        """
        checkpoint_path = Path(checkpoint_dir).resolve()
        checkpoint_path = checkpoint_path / f"checkpoint_{step}"

        # Create checkpoint directory if it doesn't exist
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Split the agent into graphdef and state
        _, state = nnx.split(self)

        # Move all arrays to CPU to ensure device-agnostic checkpoints
        # This prevents device placement issues when loading on different hardware
        cpu_device = jax.devices('cpu')[0]
        state_cpu = jax.tree.map(
            lambda x: jax.device_put(x, cpu_device) if isinstance(x, jax.Array) else x,
            state
        )

        # Save the state using Orbax
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint_path / 'agent_state', state_cpu)

        # Save metadata (agent class name and config)
        metadata = {
            'agent_class_name': self.get_class_name(),
            'agent_config': self.get_config()
        }

        with open(checkpoint_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str, step: int, key: chex.PRNGKey) -> "BaseAgent":
        """Load agent checkpoint from disk using Orbax.

        This method can be called from BaseAgent or any subclass:
        - If called from BaseAgent: Uses metadata to determine and instantiate correct subclass
        - If called from subclass: Validates metadata matches the subclass

        Best practices followed:
        - Device-agnostic: Uses SingleDeviceSharding to load on current default device
        - Automatic device placement: Arrays are placed on the default JAX device
        - Cross-device compatible: Works whether checkpoint was saved on CPU/CUDA/TPU

        Args:
            checkpoint_dir: Directory containing checkpoint
            step: step number for checkpoint naming
            key: JAX random key for initialization

        Returns:
            Loaded agent instance of the appropriate subclass

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If metadata is invalid or class mismatch detected
        """
        from agents import get_agent_class_from_name, get_agent_name_from_class

        checkpoint_path = Path(checkpoint_dir).resolve()
        checkpoint_path = checkpoint_path / f"checkpoint_{step}"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Load metadata
        metadata_path = checkpoint_path / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Checkpoint metadata not found at {metadata_path}. "
                f"This checkpoint may have been created with an older version."
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        agent_class_name = metadata['agent_class_name']
        agent_config = metadata['agent_config']

        # Determine which class to instantiate
        if cls is BaseAgent:
            # Called from BaseAgent: use metadata to get correct subclass
            target_cls = get_agent_class_from_name(agent_class_name)
        else:
            # Called from specific subclass: validate it matches metadata
            expected_name = get_agent_name_from_class(cls)
            if expected_name != agent_class_name:
                raise ValueError(
                    f"Class mismatch: checkpoint is for '{agent_class_name}' "
                    f"but load_checkpoint was called from '{expected_name}'. "
                    f"Use BaseAgent.load_checkpoint() for automatic class detection."
                )
            target_cls = cls

        # Create abstract model with config for restoration
        abstract_agent = nnx.eval_shape(lambda: target_cls(key, **agent_config))
        graphdef, abstract_state = nnx.split(abstract_agent)

        # Prepare restore args with explicit sharding for current device
        # This ensures checkpoint can be loaded on any device (CPU/CUDA/TPU)
        default_device = jax.devices()[0]
        sharding = jax.sharding.SingleDeviceSharding(default_device)

        # Create restore args: ArrayRestoreArgs for arrays (ShapeDtypeStruct), None for others
        def create_restore_args(x):
            if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
                return ocp.ArrayRestoreArgs(sharding=sharding)
            return None  # Let Orbax use default restoration for non-arrays

        restore_args = jax.tree.map(create_restore_args, abstract_state)

        # Restore the checkpoint with explicit sharding
        checkpointer = ocp.PyTreeCheckpointer()
        restored_state = checkpointer.restore(
            checkpoint_path / 'agent_state',
            item=abstract_state,
            restore_args=restore_args
        )

        # Merge graphdef and restored state to create the agent
        agent = nnx.merge(graphdef, restored_state)

        return agent


class StatelessAgent(BaseAgent, abc.ABC):
    """Abstract base class for stateless RL agents (MLP, CNN).

    Stateless agents process each observation independently without maintaining
    hidden state across timesteps. This is the standard RL agent pattern.

    Subclasses must implement:
    - get_value: Value function estimation
    - get_action: Action sampling
    - get_action_and_value: Combined action and value computation
    - get_action_distribution: Action distribution from policy
    """

    @abc.abstractmethod
    def get_value(self, observations: env_types.Observation, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Compute state value using the value network.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            key: Optional JAX random key for stochastic operations

        Returns:
            Value estimates of shape (batch_size,)
        """
        ...

    @abc.abstractmethod
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey) -> env_types.Action:
        """Sample action from policy network.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            key: JAX random key for sampling

        Returns:
            Sampled actions of shape (batch_size,)
        """
        ...

    @abc.abstractmethod
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey
        ) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            key: JAX random key for sampling

        Returns:
            Tuple of (action, log_prob, value) arrays of shape (batch_size,)
        """
        ...

    @abc.abstractmethod
    def get_action_distribution(
        self, observations: env_types.Observation, key: Optional[chex.PRNGKey] = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            key: Optional JAX random key for stochastic operations

        Returns:
            distrax.Distribution of shape (batch_size,)
        """
        ...

    @abc.abstractmethod
    def get_distribution_and_value(
        self, observations: env_types.Observation, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[distrax.Distribution, chex.Array]:
        """Get action distribution and value simultaneously.

        This is the main method used during PPO training to evaluate trajectories.
        More efficient than calling get_action_distribution and get_value separately.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            key: Optional JAX random key for stochastic operations

        Returns:
            Tuple of (distribution, values) where values has shape (batch_size,)
        """
        ...


class StatefulAgent(BaseAgent, abc.ABC):
    """Abstract base class for stateful RL agents (LSTM, GRU).

    Stateful agents maintain hidden state across timesteps, allowing them to
    remember past observations. The training script is responsible for managing
    hidden states and resetting them at episode boundaries.

    Subclasses must implement:
    - initialize_carry: Initialize hidden states
    - get_carry_spec: Get shape/dtype specification for preallocation
    - get_value: Value estimation with hidden state
    - get_action: Action sampling with hidden state
    - get_action_and_value: Combined computation with hidden states
    - get_action_distribution: Action distribution with hidden state
    """

    @abc.abstractmethod
    def initialize_carry(self, batch_size: int) -> Any:
        """Initialize hidden states for a batch.

        Args:
            batch_size: Number of parallel environments/sequences

        Returns:
            Initial hidden state (structure depends on agent type)
        """
        ...

    def get_carry_spec(self, batch_size: int) -> Any:
        """Get shape/dtype specification of carries for preallocation.

        This returns a pytree of jax.ShapeDtypeStruct that describes the
        structure of carries without actually allocating them. Useful for
        buffer preallocation.

        Args:
            batch_size: Number of parallel environments/sequences

        Returns:
            Pytree of ShapeDtypeStruct matching initialize_carry structure

        Note:
            Default implementation uses eval_shape on initialize_carry.
            Subclasses can override for more efficient implementations.
        """
        return jax.eval_shape(lambda: self.initialize_carry(batch_size))

    @abc.abstractmethod
    def get_value(
        self, observations: env_types.Observation, carry: Any, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[chex.Array, Any]:
        """Compute state value with hidden state.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            carry: Hidden state from previous timestep with shape (batch_size, *carry_shape)
                  The carry is batched - one hidden state per sample in the batch
            key: Optional JAX random key for stochastic operations

        Returns:
            Tuple of (values, new_carry) where new_carry has shape (batch_size, *carry_shape)
        """
        ...

    @abc.abstractmethod
    def get_action(
        self, observations: env_types.Observation, carry: Any, key: chex.PRNGKey
    ) -> Tuple[env_types.Action, Any]:
        """Sample action with hidden state.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            carry: Hidden state from previous timestep with shape (batch_size, *carry_shape)
                  The carry is batched - one hidden state per sample in the batch
            key: JAX random key for sampling

        Returns:
            Tuple of (actions, new_carry) where new_carry has shape (batch_size, *carry_shape)
        """
        ...

    @abc.abstractmethod
    def get_action_and_value(
        self,
        observations: env_types.Observation,
        carry: Any,
        key: chex.PRNGKey
    ) -> Tuple[env_types.Action, chex.Array, chex.Array, Any]:
        """Sample action and compute value with hidden state.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            carry: Hidden state from previous timestep with shape (batch_size, *carry_shape)
                  The carry is batched - one hidden state per sample in the batch
            key: JAX random key for sampling

        Returns:
            Tuple of (actions, log_probs, values, new_carry) where new_carry has shape (batch_size, *carry_shape)
        """
        ...

    @abc.abstractmethod
    def get_action_distribution(
        self, observations: env_types.Observation, carry: Any, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[distrax.Distribution, Any]:
        """Get action distribution with hidden state.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            carry: Hidden state from previous timestep with shape (batch_size, *carry_shape)
                  The carry is batched - one hidden state per sample in the batch
            key: Optional JAX random key for stochastic operations

        Returns:
            Tuple of (distribution, new_carry) where new_carry has shape (batch_size, *carry_shape)
        """
        ...

    @abc.abstractmethod
    def get_distribution_and_value(
        self, observations: env_types.Observation, carry: Any, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[distrax.Distribution, chex.Array, Any]:
        """Get action distribution and value with hidden state.

        This is the main method used during PPO training to evaluate trajectories.
        More efficient than calling get_action_distribution and get_value separately,
        and ensures both policy and critic carries are properly updated.

        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            carry: Hidden state from previous timestep with shape (batch_size, *carry_shape)
                  The carry is batched - one hidden state per sample in the batch
            key: Optional JAX random key for stochastic operations

        Returns:
            Tuple of (distribution, values, new_carry) where:
            - values has shape (batch_size,)
            - new_carry has shape (batch_size, *carry_shape)
        """
        ...