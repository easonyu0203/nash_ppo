"""Embedding-MLP agent for discrete token observations."""

from typing import Union, Tuple
import jax
import chex

from agents.base import ActorCriticAgent, ActionSpaceType
from agents.networks.feature_extractors import EmbeddingMLPFeatureExtractor


class EmbeddingMLPAgent(ActorCriticAgent):
    """Embedding-MLP agent for discrete token observations.

    This agent first embeds discrete token observations into a continuous space,
    then processes them through MLP layers. Suitable for environments with discrete
    observation spaces (e.g., copying tasks, language models).

    Architecture:
        discrete token -> Embedding -> MLP -> policy/value

    Args:
        key: JAX random key
        vocab_size: Size of vocabulary (number of distinct tokens)
        output_dim: Output action dimension(s).
                   - int: for Discrete or Continuous action space
                   - Tuple[int, ...]: for MultiDiscrete action space
        action_space_type: Type of action space - ActionSpaceType enum or string
                          ("discrete", "multi_discrete", or "continuous")
        embedding_dim: Dimension of embedding vectors (default: 32)
        mlp_dim: Hidden layer dimension (default: 64)
        num_hidden_layers: Number of MLP hidden layers (default: 2)

    Example:
        >>> agent = EmbeddingMLPAgent(
        ...     key,
        ...     vocab_size=10,
        ...     output_dim=10,
        ...     action_space_type="discrete",
        ...     embedding_dim=32,
        ...     mlp_dim=64
        ... )
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        vocab_size: int,
        output_dim: Union[int, Tuple[int, ...]],
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.DISCRETE,
        embedding_dim: int = 32,
        mlp_dim: int = 64,
        num_hidden_layers: int = 2,
    ):
        """Initialize Embedding-MLP agent."""
        # Split keys for policy and critic networks
        key_policy, key_critic = jax.random.split(key)

        # Create separate embedding-MLP extractors for policy and critic
        policy_extractor = EmbeddingMLPFeatureExtractor(
            key_policy, vocab_size, embedding_dim, mlp_dim, num_hidden_layers
        )
        critic_extractor = EmbeddingMLPFeatureExtractor(
            key_critic, vocab_size, embedding_dim, mlp_dim, num_hidden_layers
        )

        # Initialize parent actor-critic agent
        super().__init__(
            key=key,
            policy_extractor=policy_extractor,
            critic_extractor=critic_extractor,
            action_dim=output_dim,
            action_space_type=action_space_type,
        )
