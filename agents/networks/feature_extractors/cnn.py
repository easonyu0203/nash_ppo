"""CNN feature extractor for image observations."""

from typing import Tuple, Sequence, Optional
import jax.numpy as jnp
import chex
from flax import nnx

from agents.networks.feature_extractors.base import FeatureExtractor


class CNNFeatureExtractor(FeatureExtractor):
    """CNN-based feature extractor for image observations."""

    def __init__(
        self,
        key: chex.PRNGKey,
        input_shape: Tuple[int, ...],
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        pool_size: int = 2,
        normalize: bool = False,
        normalization_factor: float = 255.0
    ):
        """Initialize CNN feature extractor.

        Architecture:
        - Each CNN layer: Conv(kernel_size x kernel_size, SAME) -> ReLU -> MaxPool
        - Final: Global MaxPool over spatial dimensions -> feature vector

        Args:
            key: JAX random key
            input_shape: Shape of input images (H, W, C) - channels last
            cnn_channels: Number of channels for each CNN layer
            kernel_size: Kernel size for all conv layers (default: 3)
            pool_size: Max pooling size and stride (default: 2)
            normalize: Whether to normalize input images
            normalization_factor: Value to divide by for normalization (default: 255.0)
        """
        rngs = nnx.Rngs(key)

        self.normalize = normalize
        self.normalization_factor = normalization_factor
        self.pool_size = pool_size

        # Build CNN layers with max pooling after each
        conv_layers_list = []
        in_channels = input_shape[-1]  # Channels-last format (H, W, C)

        for out_channels in cnn_channels:
            conv_layers_list.append(
                nnx.Conv(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(1, 1),
                    padding='SAME',
                    rngs=rngs
                )
            )
            in_channels = out_channels

        # Use nnx.List for proper pytree handling
        self.conv_layers = nnx.List(conv_layers_list)
        self._output_dim = in_channels

    def __call__(self, observations: chex.Array, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Extract features from image observations.

        Args:
            observations: Image tensor of shape (batch_size, H, W, C)
            key: Optional JAX random key for stochastic operations

        Returns:
            Feature tensor of shape (batch_size, cnn_channels[-1])
        """
        # Convert to float32
        x = observations.astype(jnp.float32)

        # Normalize if specified
        if self.normalize:
            x = x / self.normalization_factor

        # Process through CNN layers with max pooling
        for conv in self.conv_layers:
            x = conv(x)
            x = nnx.relu(x)
            # Max pool with window (pool_size, pool_size) and stride (pool_size, pool_size)
            x = nnx.max_pool(
                x,
                window_shape=(self.pool_size, self.pool_size),
                strides=(self.pool_size, self.pool_size),
                padding='VALID'
            )

        # Global max pooling over remaining spatial dimensions
        # x shape: (batch_size, H', W', C)
        # Output: (batch_size, C)
        x = jnp.max(x, axis=(1, 2))

        return x

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim
