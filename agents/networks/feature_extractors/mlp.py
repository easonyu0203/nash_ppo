"""MLP feature extractor for vector observations."""

from typing import Optional
import jax.numpy as jnp
import chex
from flax import nnx

from agents.networks.feature_extractors.base import FeatureExtractor


class MLPFeatureExtractor(FeatureExtractor):
    """MLP-based feature extractor for vector observations."""

    def __init__(
        self,
        key: chex.PRNGKey,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1
    ):
        """Initialize MLP feature extractor.

        Args:
            key: JAX random key
            input_dim: Input observation dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers (default: 1)
        """
        rngs = nnx.Rngs(key)
        layers = [nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu]

        for _ in range(num_layers - 1):
            layers += [nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu]

        self.mlp = nnx.Sequential(*layers)
        self._output_dim = hidden_dim

    def __call__(self, observations: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Extract features from vector observations.

        Args:
            observations: Observation tensor of shape (batch_size, *obs_shape)
            key: Optional JAX random key for stochastic operations

        Returns:
            Feature tensor of shape (batch_size, hidden_dim)
        """
        flattened = observations.reshape(observations.shape[0], -1).astype(jnp.float32)
        return self.mlp(flattened)

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim
