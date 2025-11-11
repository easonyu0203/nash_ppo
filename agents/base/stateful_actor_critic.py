"""Stateful Actor-Critic agent for recurrent networks with hidden states."""

from typing import Tuple, Union, Any
import jax
import chex
import distrax
from flax import nnx

from agents.base.agent import StatefulAgent
from agents.base.types import ActionSpaceType
from agents.networks.feature_extractors import RecurrentFeatureExtractor
from agents.networks.policy_heads import create_policy_head
from agents.utils import layer_init
import envs.mytypes as env_types


class StatefulActorCriticAgent(StatefulAgent):
    """Actor-critic agent with recurrent networks that maintain hidden state.

    This agent extends the standard actor-critic pattern to handle recurrent
    feature extractors (LSTM, GRU) that maintain hidden states across timesteps.

    Architecture:
        obs -> policy_extractor(obs, carry) -> features, new_carry -> policy_head -> action
        obs -> critic_extractor(obs, carry) -> features, new_carry -> value_head -> value

    The training script is responsible for:
    - Managing hidden states across rollouts
    - Resetting states at episode boundaries
    - Handling sequences properly
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        policy_extractor: RecurrentFeatureExtractor,
        critic_extractor: RecurrentFeatureExtractor,
        action_dim: Union[int, Tuple[int, ...]],
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.DISCRETE,
    ):
        """Initialize stateful actor-critic agent.

        Args:
            key: JAX random key
            policy_extractor: Recurrent feature extractor for policy network
            critic_extractor: Recurrent feature extractor for value network
            action_dim: Action dimension(s)
            action_space_type: Type of action space
        """
        rngs = nnx.Rngs(key)

        # Normalize action_space_type to enum
        if isinstance(action_space_type, str):
            self.action_space_type = ActionSpaceType(action_space_type)
        else:
            self.action_space_type = action_space_type

        # Store extractors and dimensions
        self.policy_extractor = policy_extractor
        self.critic_extractor = critic_extractor
        self.action_dim = action_dim

        # Create policy head based on action space type
        self.policy_head = create_policy_head(
            self.action_space_type,
            self.policy_extractor.output_dim,
            action_dim,
            rngs
        )

        # Create value head (same for all action spaces)
        self.value_head = nnx.Linear(self.critic_extractor.output_dim, 1, rngs=rngs)

        # Initialize all parameters
        self._initialize_weights(rngs)

    def _initialize_weights(self, rngs: nnx.Rngs):
        """Initialize network weights with appropriate std."""
        # Initialize all layers with default std
        layer_init(self, rngs.param())

        # Initialize policy head with smaller std for stability
        layer_init(self.policy_head, rngs.param(), std=0.01)

    # ===== Hidden state management =====

    def initialize_carry(self, batch_size: int) -> Tuple[Any, Any]:
        """Initialize hidden states for both policy and critic networks.

        Args:
            batch_size: Number of parallel environments/sequences

        Returns:
            Tuple of (policy_carry, critic_carry)
        """
        policy_carry = self.policy_extractor.initialize_carry(batch_size)
        critic_carry = self.critic_extractor.initialize_carry(batch_size)
        return policy_carry, critic_carry

    # ===== Core agent interface (stateful versions) =====

    @jax.jit
    def get_value(
        self, observations: env_types.Observation, carry: Any
    ) -> Tuple[chex.Array, Any]:
        """Compute state value estimate with hidden state.

        Args:
            observations: Observation tensor
            carry: Full carry tuple (policy_carry, critic_carry)

        Returns:
            Tuple of (values, new_carry) where new_carry is (policy_carry, new_critic_carry)
        """
        policy_carry, critic_carry = carry
        features, new_critic_carry = self.critic_extractor(observations, critic_carry)
        values = self.value_head(features).squeeze(-1)
        return values, (policy_carry, new_critic_carry)

    @jax.jit
    def get_action_distribution(
        self, observations: env_types.Observation, carry: Any
    ) -> Tuple[distrax.Distribution, Any]:
        """Get action distribution from policy network with hidden state.

        Args:
            observations: Observation tensor
            carry: Full carry tuple (policy_carry, critic_carry)

        Returns:
            Tuple of (distribution, new_carry) where new_carry is (new_policy_carry, critic_carry)
        """
        policy_carry, critic_carry = carry
        features, new_policy_carry = self.policy_extractor(observations, policy_carry)
        dist = self.policy_head(features)
        return dist, (new_policy_carry, critic_carry)

    @jax.jit
    def get_action(
        self, observations: env_types.Observation, carry: Any, key: chex.PRNGKey
    ) -> Tuple[env_types.Action, Any]:
        """Sample an action from the current policy with hidden state.

        Args:
            observations: Observation tensor
            carry: Full carry tuple (policy_carry, critic_carry)
            key: JAX random key for sampling

        Returns:
            Tuple of (actions, new_carry) where new_carry is (new_policy_carry, critic_carry)
        """
        dist, new_carry = self.get_action_distribution(observations, carry)
        actions = dist.sample(seed=key)
        return actions, new_carry

    @jax.jit
    def get_action_and_value(
        self,
        observations: env_types.Observation,
        carry: Any,
        key: chex.PRNGKey
    ) -> Tuple[env_types.Action, chex.Array, chex.Array, Any]:
        """Sample action and compute value with hidden states.

        This is the main method used during rollouts.

        Args:
            observations: Observation tensor
            carry: Full carry tuple (policy_carry, critic_carry)
            key: JAX random key for sampling

        Returns:
            Tuple of (actions, log_probs, values, new_carry)
                where new_carry is (new_policy_carry, new_critic_carry)
        """
        policy_carry, critic_carry = carry

        # Get action distribution and sample
        policy_features, new_policy_carry = self.policy_extractor(observations, policy_carry)
        dist = self.policy_head(policy_features)
        actions, log_probs = dist.sample_and_log_prob(seed=key)

        # Get value estimate
        critic_features, new_critic_carry = self.critic_extractor(observations, critic_carry)
        values = self.value_head(critic_features).squeeze(-1)

        new_carry = (new_policy_carry, new_critic_carry)
        return actions, log_probs, values, new_carry

    @jax.jit
    def get_distribution_and_value(
        self, observations: env_types.Observation, carry: Any
    ) -> Tuple[distrax.Distribution, chex.Array, Any]:
        """Get action distribution and value with hidden states.

        This is the main method used during PPO training to evaluate trajectories.
        Ensures both policy and critic carries are properly updated.

        Args:
            observations: Observation tensor
            carry: Full carry tuple (policy_carry, critic_carry)

        Returns:
            Tuple of (distribution, values, new_carry)
                where new_carry is (new_policy_carry, new_critic_carry)
        """
        policy_carry, critic_carry = carry

        # Get action distribution
        policy_features, new_policy_carry = self.policy_extractor(observations, policy_carry)
        dist = self.policy_head(policy_features)

        # Get value estimate
        critic_features, new_critic_carry = self.critic_extractor(observations, critic_carry)
        values = self.value_head(critic_features).squeeze(-1)

        new_carry = (new_policy_carry, new_critic_carry)
        return dist, values, new_carry
