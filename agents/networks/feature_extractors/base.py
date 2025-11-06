"""Base feature extractor interface."""

from abc import ABC, abstractmethod
import chex
from flax import nnx


class FeatureExtractor(nnx.Module, ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def __call__(self, observations: chex.Array) -> chex.Array:
        """Extract features from observations.

        Args:
            observations: Input observations

        Returns:
            Feature tensor
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of output features."""
        ...
