"""Policy head networks for different action space types."""

from collections.abc import Iterable
from typing import Union, Tuple
from abc import ABC, abstractmethod
import jax.numpy as jnp
import chex
import distrax
from flax import nnx

from agents.base.types import ActionSpaceType


class PolicyHead(nnx.Module, ABC):
    """Abstract base class for policy heads."""

    @abstractmethod
    def __call__(self, features: chex.Array) -> distrax.Distribution:
        """Convert features to action distribution.

        Args:
            features: Feature tensor from feature extractor

        Returns:
            Action distribution
        """
        ...


class DiscretePolicyHead(PolicyHead):
    """Policy head for discrete action spaces."""

    def __init__(self, feature_dim: int, action_dim: int, rngs: nnx.Rngs):
        """Initialize discrete policy head.

        Args:
            feature_dim: Dimension of input features
            action_dim: Number of discrete actions
            rngs: Random number generator state
        """
        self.linear = nnx.Linear(feature_dim, action_dim, rngs=rngs)

    def __call__(self, features: chex.Array) -> distrax.Distribution:
        """Get categorical distribution over actions.

        Args:
            features: Feature tensor of shape (batch_size, feature_dim)

        Returns:
            Categorical distribution over actions
        """
        logits = self.linear(features)
        return distrax.Categorical(logits=logits)


class MultiDiscretePolicyHead(PolicyHead):
    """Policy head for multi-discrete action spaces."""

    def __init__(
        self,
        feature_dim: int,
        action_dims: Tuple[int, ...],
        rngs: nnx.Rngs
    ):
        """Initialize multi-discrete policy head.

        Args:
            feature_dim: Dimension of input features
            action_dims: Tuple of action dimensions for each discrete space
            rngs: Random number generator state
        """
        self.heads = nnx.List([
            nnx.Linear(feature_dim, n_actions, rngs=rngs)
            for n_actions in action_dims
        ])

    def __call__(self, features: chex.Array) -> distrax.Distribution:
        """Get independent categorical distributions.

        Args:
            features: Feature tensor of shape (batch_size, feature_dim)

        Returns:
            Independent categorical distributions
        """
        logits = jnp.stack([head(features) for head in self.heads], axis=1)
        return distrax.Independent(
            distrax.Categorical(logits=logits),
            reinterpreted_batch_ndims=1
        )


class ContinuousPolicyHead(PolicyHead):
    """Policy head for continuous action spaces."""

    def __init__(self, feature_dim: int, action_dim: int, rngs: nnx.Rngs):
        """Initialize continuous policy head.

        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of continuous action space
            rngs: Random number generator state
        """
        self.mean_layer = nnx.Linear(feature_dim, action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, features: chex.Array) -> distrax.Distribution:
        """Get normal distribution for continuous actions.

        Args:
            features: Feature tensor of shape (batch_size, feature_dim)

        Returns:
            Independent normal distribution
        """
        mean = self.mean_layer(features)
        std = jnp.exp(self.log_std)
        return distrax.Independent(
            distrax.Normal(mean, std),
            reinterpreted_batch_ndims=1
        )


def create_policy_head(
    action_space_type: ActionSpaceType,
    feature_dim: int,
    action_dim: Union[int, Tuple[int, ...]],
    rngs: nnx.Rngs
) -> PolicyHead:
    """Factory function to create appropriate policy head.

    Args:
        action_space_type: Type of action space
        feature_dim: Dimension of input features
        action_dim: Action dimension(s) - int for discrete/continuous, tuple for multi-discrete
        rngs: Random number generator state

    Returns:
        Appropriate policy head instance

    Raises:
        ValueError: If action_dim type doesn't match action_space_type
    """
    if action_space_type == ActionSpaceType.DISCRETE:
        if not isinstance(action_dim, int):
            raise ValueError(f"Discrete action space requires int action_dim, got {type(action_dim)}")
        return DiscretePolicyHead(feature_dim, action_dim, rngs)

    elif action_space_type == ActionSpaceType.MULTI_DISCRETE:
        if not isinstance(action_dim, Iterable):
            raise ValueError(f"Multi-discrete action space requires iterable action_dim, got {type(action_dim)}")
        return MultiDiscretePolicyHead(feature_dim, action_dim, rngs)

    elif action_space_type == ActionSpaceType.CONTINUOUS:
        if not isinstance(action_dim, int):
            raise ValueError(f"Continuous action space requires int action_dim, got {type(action_dim)}")
        return ContinuousPolicyHead(feature_dim, action_dim, rngs)

    else:
        raise ValueError(f"Unknown action space type: {action_space_type}")
