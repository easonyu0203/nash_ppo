"""Base Actor-Critic agent with shared policy/value logic."""

from typing import Tuple, Union, Optional
import jax
import jax.numpy as jnp
import chex
import distrax
from flax import nnx

from agents.base.agent import StatelessAgent
from agents.base.types import ActionSpaceType
from agents.networks.feature_extractors import FeatureExtractor
from agents.networks.policy_heads import create_policy_head
from agents.utils import layer_init
import envs.mytypes as env_types


class ActorCriticAgent(StatelessAgent):
    """Base actor-critic agent with separate policy and value networks.

    This class implements the common actor-critic architecture used by both
    MLP and CNN-based agents. It uses composition to combine:
    - Feature extractors (policy and critic)
    - Policy head (action space specific)
    - Value head (shared across all action spaces)

    The key insight is that after feature extraction, all agents follow the same
    pattern regardless of observation type (vector vs image).
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        policy_extractor: FeatureExtractor,
        critic_extractor: FeatureExtractor,
        action_dim: Union[int, Tuple[int, ...]],
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.DISCRETE,
    ):
        """Initialize actor-critic agent.

        Args:
            key: JAX random key
            policy_extractor: Feature extractor for policy network
            critic_extractor: Feature extractor for value network
            action_dim: Action dimension(s) - int for discrete/continuous, tuple for multi-discrete
            action_space_type: Type of action space

        Note: The feature extractors are created externally and passed in,
              allowing different agents to use different extractors (MLP, CNN, etc.)
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

    # ===== Core agent interface =====

    @jax.jit
    def get_value(self, observations: env_types.Observation, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Compute state value estimate.

        Args:
            observations: Observation tensor
            key: Optional JAX random key for stochastic operations

        Returns:
            Value estimates of shape (batch_size,)
        """
        features = self.critic_extractor(observations, key)
        return self.value_head(features).squeeze(-1)

    @jax.jit
    def get_action_distribution(
        self, observations: env_types.Observation, key: Optional[chex.PRNGKey] = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network.

        Args:
            observations: Observation tensor
            key: Optional JAX random key for stochastic operations

        Returns:
            Action distribution (type depends on action space)
        """
        features = self.policy_extractor(observations, key)
        return self.policy_head(features)

    @jax.jit
    def get_action(
        self, observations: env_types.Observation, key: chex.PRNGKey
    ) -> env_types.Action:
        """Sample an action from the current policy.

        Args:
            observations: Observation tensor
            key: JAX random key for sampling

        Returns:
            Sampled actions
        """
        return self.get_action_distribution(observations).sample(seed=key)

    @jax.jit
    def get_action_and_value(
        self, observations: env_types.Observation, key: chex.PRNGKey
    ) -> Tuple[env_types.Action, chex.Array, chex.Array]:
        """Sample action and compute log probability and value simultaneously.

        This is more efficient than calling get_action and get_value separately
        as it avoids redundant forward passes.

        Args:
            observations: Observation tensor
            key: JAX random key for sampling

        Returns:
            Tuple of (actions, log_probs, values)
        """
        # Get action distribution and sample
        dist = self.get_action_distribution(observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)

        # Get value estimate
        values = self.get_value(observations)

        return actions, log_probs, values

    @jax.jit
    def get_distribution_and_value(
        self, observations: env_types.Observation, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[distrax.Distribution, chex.Array]:
        """Get action distribution and value simultaneously.

        This is the main method used during PPO training to evaluate trajectories.
        More efficient than calling get_action_distribution and get_value separately.

        Args:
            observations: Observation tensor
            key: Optional JAX random key for stochastic operations

        Returns:
            Tuple of (distribution, values)
        """
        # Split key for policy and critic extractors if provided
        if key is not None:
            policy_key, critic_key = jax.random.split(key)
        else:
            policy_key = critic_key = None

        # Get action distribution
        policy_features = self.policy_extractor(observations, policy_key)
        dist = self.policy_head(policy_features)

        # Get value estimate
        critic_features = self.critic_extractor(observations, critic_key)
        values = self.value_head(critic_features).squeeze(-1)

        return dist, values
