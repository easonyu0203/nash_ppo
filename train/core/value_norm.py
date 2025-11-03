"""
Value normalization for stabilizing RL training.

Adapted from: https://github.com/marlbenchmark/on-policy
Converted from PyTorch to JAX/Flax.
"""

import jax.numpy as jnp
from flax import nnx
import chex


class ValueNorm(nnx.Module):
    """
    Normalize value function targets using running statistics.

    Maintains exponential moving average of mean and variance to normalize
    value targets to zero mean and unit variance, stabilizing training.

    Key operations:
    - update(): Update running statistics with new batch of returns
    - normalize(): Transform returns to normalized space (for value loss)
    - denormalize(): Transform normalized predictions back to original scale (for advantages)
    """

    def __init__(
        self,
        norm_axes: int = 1,
        beta: float = 0.99999,
        per_element_update: bool = False,
        epsilon: float = 1e-5
    ):
        """
        Initialize value normalization for scalar values.

        Args:
            norm_axes: Number of axes to normalize over (default: 1)
            beta: Decay factor for exponential moving average (closer to 1 = slower adaptation)
            per_element_update: If True, adjust decay based on batch size
            epsilon: Small constant for numerical stability
        """
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        # Running statistics (non-trainable)
        # Use Variable with collection='' to mark as mutable state (not parameters)
        # Always use shape=1 for scalar value normalization
        self.running_mean = nnx.Variable(jnp.zeros(1))
        self.running_mean_sq = nnx.Variable(jnp.zeros(1))
        self.debiasing_term = nnx.Variable(jnp.zeros(()))

    def reset_parameters(self):
        """Reset all statistics to zero"""
        self.running_mean.value = jnp.zeros(1)
        self.running_mean_sq.value = jnp.zeros(1)
        self.debiasing_term.value = jnp.zeros(())

    def running_mean_var(self) -> tuple[chex.Array, chex.Array]:
        """
        Compute debiased mean and variance from running statistics.

        Returns:
            Tuple of (mean, variance) with variance clipped to minimum of 1e-2
        """
        debiased_mean = self.running_mean.value / jnp.maximum(self.debiasing_term.value, self.epsilon)
        debiased_mean_sq = self.running_mean_sq.value / jnp.maximum(self.debiasing_term.value, self.epsilon)
        debiased_var = jnp.maximum(debiased_mean_sq - debiased_mean ** 2, 1e-2)
        return debiased_mean, debiased_var

    def update(self, input_vector: chex.Array):
        """
        Update running statistics with new batch of data.

        This should be called BEFORE normalize() during training.

        Args:
            input_vector: Batch of values to update statistics with
                         Shape: (batch_size, ...) or any shape with batch dim first
        """
        # Ensure input is float32
        if input_vector.dtype != jnp.float32:
            input_vector = input_vector.astype(jnp.float32)

        # Compute batch statistics
        batch_mean = jnp.mean(input_vector, axis=tuple(range(self.norm_axes)))
        batch_sq_mean = jnp.mean(input_vector ** 2, axis=tuple(range(self.norm_axes)))

        # Compute weight for exponential moving average
        if self.per_element_update:
            batch_size = jnp.prod(jnp.array(input_vector.shape[:self.norm_axes]))
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        # Update running statistics (in-place mutation)
        self.running_mean.value = self.running_mean.value * weight + batch_mean * (1.0 - weight)
        self.running_mean_sq.value = self.running_mean_sq.value * weight + batch_sq_mean * (1.0 - weight)
        self.debiasing_term.value = self.debiasing_term.value * weight + (1.0 - weight)

    def normalize(self, input_vector: chex.Array) -> chex.Array:
        """
        Normalize input using running statistics.

        Use this to normalize target returns before computing value loss.

        Args:
            input_vector: Values to normalize

        Returns:
            Normalized values with ~0 mean and ~1 std
        """
        # Ensure input is float32
        if input_vector.dtype != jnp.float32:
            input_vector = input_vector.astype(jnp.float32)

        mean, var = self.running_mean_var()

        # For scalar values (input_shape=1), mean and var are already scalars
        out = (input_vector - mean) / jnp.sqrt(var)
        return out

    def denormalize(self, input_vector: chex.Array) -> chex.Array:
        """
        Transform normalized data back to original distribution.

        Use this to denormalize value predictions when computing advantages.

        Args:
            input_vector: Normalized values

        Returns:
            Values in original scale
        """
        # Ensure input is float32
        if input_vector.dtype != jnp.float32:
            input_vector = input_vector.astype(jnp.float32)

        mean, var = self.running_mean_var()

        # For scalar values (input_shape=1), mean and var are already scalars
        out = input_vector * jnp.sqrt(var) + mean
        return out
