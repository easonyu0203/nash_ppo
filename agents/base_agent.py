from pathlib import Path
from typing import Optional, Tuple
from flax import nnx
import chex
import abc
import distrax
import orbax.checkpoint as ocp
import json

import envs.mytypes as env_types
from agents.configurable_agent import ConfigurableAgent

class BaseAgent(nnx.Module, ConfigurableAgent, abc.ABC):

    @abc.abstractmethod
    def __init__(self, key: chex.PRNGKey, **kwargs):
        pass

    @abc.abstractmethod
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value using the value network.
        
        Args:
            state: Observation tensor of shape (batch_size, *observation_shape)
            
        Returns:
            Value estimates of shape (batch_size,)
        """
        pass
    
    @abc.abstractmethod
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None) -> env_types.Action:
        """Sample action from policy network.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            Sampled actions of shape (batch_size, )
        """
        pass
    
    @abc.abstractmethod
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None
        ) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously.
        
        Args:
            observations (batch_size, *observation_shape): Observation tensor
            key (): JAX random key for sampling
            action_masks (batch_size, num_actions): Binary mask for valid actions
        
        Returns:
            Tuple of (action, log_prob, value) arrays of shape (batch_size,)
        """
        pass
    
    @abc.abstractmethod
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: Optional[chex.Array] = None
    ) -> distrax.Distribution:
        """Get action distribution
        
        Args:
            observations: Observation tensor of shape (batch_size, *observation_shape)
            action_masks (batch_size, num_actions): Binary mask for valid actions
            
        Returns:
            distrax.Distribution of shape (batch_size, )
        """
        pass

    def save_checkpoint(self, checkpoint_dir: str, step: int):
        """Save agent checkpoint to disk using Orbax.

        Saves three components:
        1. agent_state: Neural network parameters (Orbax format)
        2. metadata.json: Agent class name and configuration

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

        # Save the state using Orbax
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(checkpoint_path / 'agent_state', state)

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

        # Restore the checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        restored_state = checkpointer.restore(checkpoint_path / 'agent_state', abstract_state)

        # Merge graphdef and restored state to create the agent
        agent = nnx.merge(graphdef, restored_state)

        return agent