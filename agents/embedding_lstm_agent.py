"""Embedding-LSTM agent for discrete token observations with sequential memory."""

from typing import Union, Tuple
import jax
import chex

from agents.base import StatefulActorCriticAgent, ActionSpaceType
from agents.networks.feature_extractors import EmbeddingMLPFeatureExtractor, LSTMFeatureExtractor


class EmbeddingLSTMAgent(StatefulActorCriticAgent):
    """Embedding-LSTM agent for discrete token observations.

    This agent first embeds discrete token observations, processes them through MLP,
    then uses LSTM for temporal reasoning. Suitable for sequential tasks with discrete
    observations (e.g., copying tasks, language modeling).

    Architecture:
        discrete token -> Embedding -> MLP -> LSTM -> policy/value

    Args:
        key: JAX random key
        vocab_size: Size of vocabulary (number of distinct tokens)
        output_dim: Output action dimension(s)
        action_space_type: Type of action space (default: discrete)
        embedding_dim: Dimension of embedding vectors (default: 32)
        mlp_dim: MLP hidden dimension (default: 64)
        lstm_dim: LSTM hidden dimension (default: 128)
        num_mlp_layers: Number of MLP layers (default: 2)
        num_lstm_layers: Number of stacked LSTM layers (default: 1)

    Example:
        >>> agent = EmbeddingLSTMAgent(
        ...     key,
        ...     vocab_size=10,
        ...     output_dim=10,
        ...     embedding_dim=32,
        ...     mlp_dim=64,
        ...     lstm_dim=128
        ... )
        >>> carry = agent.initialize_carry(batch_size=4)
        >>> actions, log_probs, values, new_carry = agent.get_action_and_value(obs, carry, key)
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        vocab_size: int,
        output_dim: Union[int, Tuple[int, ...]],
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.DISCRETE,
        embedding_dim: int = 32,
        mlp_dim: int = 64,
        lstm_dim: int = 128,
        num_mlp_layers: int = 2,
        num_lstm_layers: int = 1,
    ):
        """Initialize Embedding-LSTM agent."""
        key_policy_emb, key_critic_emb, key_policy_lstm, key_critic_lstm = jax.random.split(key, 4)

        # Create Embedding-MLP base extractors
        policy_emb_mlp = EmbeddingMLPFeatureExtractor(
            key_policy_emb, vocab_size, embedding_dim, mlp_dim, num_mlp_layers
        )
        critic_emb_mlp = EmbeddingMLPFeatureExtractor(
            key_critic_emb, vocab_size, embedding_dim, mlp_dim, num_mlp_layers
        )

        # Wrap with LSTM
        policy_extractor = LSTMFeatureExtractor(
            key_policy_lstm, policy_emb_mlp, lstm_dim, num_lstm_layers
        )
        critic_extractor = LSTMFeatureExtractor(
            key_critic_lstm, critic_emb_mlp, lstm_dim, num_lstm_layers
        )

        # Initialize parent
        super().__init__(
            key=key,
            policy_extractor=policy_extractor,
            critic_extractor=critic_extractor,
            action_dim=output_dim,
            action_space_type=action_space_type,
        )
