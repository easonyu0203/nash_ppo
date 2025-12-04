"""Base feature extractor interface."""

from abc import ABC, abstractmethod
from typing import Optional
import chex
from flax import nnx


class FeatureExtractor(nnx.Module, ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def __call__(self, observations: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Extract features from observations.

        Args:
            observations: Input observations
            key: Optional JAX random key for stochastic operations

        Returns:
            Feature tensor
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of output features."""
        ...
