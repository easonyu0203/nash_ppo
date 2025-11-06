"""MLP-based agent for vector observations."""

from typing import Union, Tuple
import jax
import chex

from agents.base import ActorCriticAgent, ActionSpaceType
from agents.networks.feature_extractors import MLPFeatureExtractor


class MLPAgent(ActorCriticAgent):
    """MLP-based agent with support for discrete, multi-discrete, and continuous actions.

    This agent uses MLP feature extractors for both policy and value networks,
    making it suitable for vector-based observations (e.g., state vectors, positions).

    Args:
        key: JAX random key
        input_dim: Input observation dimension
        output_dim: Output action dimension(s).
                   - int: for Discrete or Continuous action space
                   - Tuple[int, ...]: for MultiDiscrete action space
        action_space_type: Type of action space - ActionSpaceType enum or string
                          ("discrete", "multi_discrete", or "continuous")
        mlp_dim: Hidden layer dimension (default: 64)
        num_hidden_layers: Number of hidden layers (default: 3)

    Example:
        >>> # Discrete action space
        >>> agent = MLPAgent(key, input_dim=10, output_dim=5, action_space_type="discrete")
        >>>
        >>> # Continuous action space
        >>> agent = MLPAgent(key, input_dim=10, output_dim=2, action_space_type="continuous")
        >>>
        >>> # Multi-discrete action space
        >>> agent = MLPAgent(key, input_dim=10, output_dim=(3, 2, 4), action_space_type="multi_discrete")
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        input_dim: int,
        output_dim: Union[int, Tuple[int, ...]],
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.DISCRETE,
        mlp_dim: int = 64,
        num_hidden_layers: int = 3,
    ):
        """Initialize MLP agent."""
        # Split keys for policy and critic networks
        key_policy, key_critic = jax.random.split(key)

        # Create separate feature extractors for policy and critic
        policy_extractor = MLPFeatureExtractor(
            key_policy, input_dim, mlp_dim, num_hidden_layers
        )
        critic_extractor = MLPFeatureExtractor(
            key_critic, input_dim, mlp_dim, num_hidden_layers
        )

        # Initialize parent actor-critic agent
        super().__init__(
            key=key,
            policy_extractor=policy_extractor,
            critic_extractor=critic_extractor,
            action_dim=output_dim,
            action_space_type=action_space_type,
        )
