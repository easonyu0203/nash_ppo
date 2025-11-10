"""Recurrent feature extractors (LSTM, GRU) that wrap base feature extractors."""

from typing import Tuple
from abc import abstractmethod
import jax
import jax.numpy as jnp
import chex
from flax import nnx

from agents.networks.feature_extractors.base import FeatureExtractor


class RecurrentFeatureExtractor(FeatureExtractor):
    """Abstract base for recurrent feature extractors with hidden state.

    Unlike standard feature extractors, recurrent extractors maintain hidden state
    that carries information across timesteps. The training script is responsible
    for managing sequences and resetting states at episode boundaries.

    Architecture pattern:
        observation -> base_extractor -> latent -> RNN -> features
    """

    @abstractmethod
    def initialize_carry(self, batch_size: int):
        """Initialize hidden state for a batch.

        Args:
            batch_size: Number of parallel environments/sequences

        Returns:
            Initial hidden state (structure depends on RNN type)
        """
        ...

    @abstractmethod
    def __call__(self, observations: chex.Array, carry) -> Tuple[chex.Array, chex.Array]:
        """Process observations with hidden state.

        Args:
            observations: Observation tensor of shape (batch_size, *obs_shape)
            carry: Hidden state from previous timestep with shape (batch_size, *carry_shape)
                  The carry is batched - one hidden state per sample in the batch

        Returns:
            Tuple of (features, new_carry) where new_carry has shape (batch_size, *carry_shape)
        """
        ...


class LSTMFeatureExtractor(RecurrentFeatureExtractor):
    """LSTM-based recurrent feature extractor with composable base extractor.

    Architecture:
        observation -> base_extractor -> latent_vec -> LSTM layers (stacked) -> features

    Supports stacking multiple LSTM layers for deeper temporal processing.
    Uses nnx.List for learnable modules and regular lists for state (carries).
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        base_extractor: FeatureExtractor,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        """Initialize LSTM feature extractor.

        Args:
            key: JAX random key
            base_extractor: Base feature extractor (e.g., MLPFeatureExtractor, CNNFeatureExtractor)
                          Processes raw observations into latent vectors
            hidden_dim: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers (default: 1)
        """
        self.base_extractor = base_extractor
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create stacked LSTM cells using nnx.List (for learnable parameters)
        keys = jax.random.split(key, num_layers)
        lstm_cells = []

        for i, k in enumerate(keys):
            rngs = nnx.Rngs(k)
            input_dim = base_extractor.output_dim if i == 0 else hidden_dim
            lstm_cells.append(nnx.LSTMCell(input_dim, hidden_dim, rngs=rngs))

        self.lstm_cells = nnx.List(lstm_cells)  # nnx.List for trainable modules
        self._output_dim = hidden_dim

    def initialize_carry(self, batch_size: int):
        """Initialize LSTM carries for all layers.

        Returns regular list (not nnx.List) since carries are state, not parameters.

        Args:
            batch_size: Batch size

        Returns:
            List of (h, c) tuples, one per layer, each of shape (batch_size, hidden_dim)
        """
        carries = []
        for _ in range(self.num_layers):
            h = jnp.zeros((batch_size, self.hidden_dim))
            c = jnp.zeros((batch_size, self.hidden_dim))
            carries.append((h, c))  # Regular tuple
        return carries  # Regular list - pytree compatible

    def __call__(self, observations: chex.Array, carry):
        """Process observations through base extractor then stacked LSTM layers.

        Args:
            observations: Observation tensor of shape (batch_size, *obs_shape)
            carry: List of LSTM carries [(h, c), ...], one per layer
                  Each (h, c) tuple has shapes (batch_size, hidden_dim)

        Returns:
            Tuple of (features, new_carries) where:
            - features: shape (batch_size, hidden_dim)
            - new_carries: list of (h, c) tuples, each with shape (batch_size, hidden_dim)
        """
        # Extract latent representation from observations
        x = self.base_extractor(observations)

        # Process through stacked LSTM layers
        new_carries = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            layer_carry = carry[i]
            layer_carry, x = lstm_cell(layer_carry, x)
            new_carries.append(layer_carry)

        return x, new_carries

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim

