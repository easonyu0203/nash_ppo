"""Embedding-based feature extractor for discrete token observations."""

from typing import Optional
import jax.numpy as jnp
import chex
from flax import nnx

from agents.networks.feature_extractors.base import FeatureExtractor


class EmbeddingMLPFeatureExtractor(FeatureExtractor):
    """Embedding + MLP feature extractor for single discrete token observations.

    Architecture:
        discrete token -> Embedding -> MLP -> features

    This is useful for environments with discrete observations (e.g., tokens, symbols)
    that should first be embedded into a continuous space before processing.

    Note: This extractor expects single token observations with shape (batch_size, 1).
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1
    ):
        """Initialize embedding-MLP feature extractor.

        Args:
            key: JAX random key
            vocab_size: Size of vocabulary (number of distinct tokens)
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Hidden layer dimension for MLP
            num_layers: Number of MLP hidden layers (default: 1)
        """
        rngs = nnx.Rngs(key)

        # Embedding layer
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embedding_dim,
            rngs=rngs
        )

        # MLP layers
        layers = [nnx.Linear(embedding_dim, hidden_dim, rngs=rngs), nnx.relu]

        for _ in range(num_layers - 1):
            layers += [nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu]

        self.mlp = nnx.Sequential(*layers)
        self._output_dim = hidden_dim

    def __call__(self, observations: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Extract features from discrete token observations.

        Args:
            observations: Integer token tensor of shape (batch_size, 1)
            key: Optional JAX random key for stochastic operations

        Returns:
            Feature tensor of shape (batch_size, hidden_dim)

        Raises:
            ValueError: If observation shape is not (batch_size, 1)
        """
        # Validate shape
        if observations.ndim != 2 or observations.shape[1] != 1:
            raise ValueError(
                f"Expected observations with shape (batch_size, 1), "
                f"got shape {observations.shape}"
            )

        # Convert to int32 and squeeze the token dimension
        tokens = observations.astype(jnp.int32).squeeze(1)  # (batch_size,)

        # Embed tokens: (batch_size, embedding_dim)
        embedded = self.embedding(tokens)

        # Pass through MLP: (batch_size, hidden_dim)
        return self.mlp(embedded)

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim
