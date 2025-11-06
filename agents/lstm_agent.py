"""LSTM-MLP agent for sequential decision making with vector observations."""

from typing import Union, Tuple
import jax
import chex

from agents.base import StatefulActorCriticAgent, ActionSpaceType
from agents.networks.feature_extractors import MLPFeatureExtractor, LSTMFeatureExtractor


class LSTMAgent(StatefulActorCriticAgent):
    """LSTM agent with MLP base for vector observations.

    Architecture:
        observation -> MLP -> latent -> LSTM -> policy/value

    Args:
        key: JAX random key
        input_dim: Input observation dimension
        output_dim: Output action dimension(s)
        action_space_type: Type of action space (default: discrete)
        mlp_dim: MLP hidden dimension (default: 64)
        lstm_dim: LSTM hidden dimension (default: 128)
        num_mlp_layers: Number of MLP layers (default: 2)
        num_lstm_layers: Number of stacked LSTM layers (default: 1)

    Example:
        >>> agent = LSTMAgent(
        ...     key,
        ...     input_dim=10,
        ...     output_dim=5,
        ...     mlp_dim=64,
        ...     lstm_dim=128
        ... )
        >>> carry = agent.initialize_carry(batch_size=4)
        >>> actions, log_probs, values, new_carry = agent.get_action_and_value(obs, carry, key)
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        input_dim: int,
        output_dim: Union[int, Tuple[int, ...]],
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.DISCRETE,
        mlp_dim: int = 64,
        lstm_dim: int = 128,
        num_mlp_layers: int = 2,
        num_lstm_layers: int = 1,
    ):
        """Initialize LSTM agent."""
        key_policy_mlp, key_critic_mlp, key_policy_lstm, key_critic_lstm = jax.random.split(key, 4)

        # Create MLP base extractors
        policy_mlp = MLPFeatureExtractor(key_policy_mlp, input_dim, mlp_dim, num_mlp_layers)
        critic_mlp = MLPFeatureExtractor(key_critic_mlp, input_dim, mlp_dim, num_mlp_layers)

        # Wrap with LSTM
        policy_extractor = LSTMFeatureExtractor(key_policy_lstm, policy_mlp, lstm_dim, num_lstm_layers)
        critic_extractor = LSTMFeatureExtractor(key_critic_lstm, critic_mlp, lstm_dim, num_lstm_layers)

        # Initialize parent
        super().__init__(
            key=key,
            policy_extractor=policy_extractor,
            critic_extractor=critic_extractor,
            action_dim=output_dim,
            action_space_type=action_space_type,
        )
