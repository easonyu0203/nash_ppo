"""CNN building blocks for agents."""

from typing import Tuple
from flax import nnx
import chex


class CNNBlock(nnx.Module):
    """CNN block with residual connections.

    Each layer in the block has a residual connection. When the number of
    channels changes, a 1x1 convolution projects the input to match dimensions.
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        in_channels: int,
        channel_sizes: Tuple[int, ...],
        kernel_size: int = 3
    ):
        """Initialize CNN block.

        Args:
            key: Random key for initialization
            in_channels: Number of input channels
            channel_sizes: Tuple of output channels for each layer, e.g., (32, 64, 128)
            kernel_size: Convolutional kernel size (default 3)
        """
        rngs = nnx.Rngs(key)

        layers = []
        projections = []

        current_channels = in_channels
        for out_channels in channel_sizes:
            # Conv layer
            conv = nnx.Conv(
                in_features=current_channels,
                out_features=out_channels,
                kernel_size=(kernel_size, kernel_size),
                padding='SAME',
                rngs=rngs
            )
            layers.append(conv)

            # Projection for residual if channels change
            if current_channels != out_channels:
                projection = nnx.Conv(
                    in_features=current_channels,
                    out_features=out_channels,
                    kernel_size=(1, 1),
                    padding='SAME',
                    rngs=rngs
                )
                projections.append(projection)
            else:
                projections.append(None)

            current_channels = out_channels

        self.layers = nnx.List(layers)
        self.projections = nnx.List(projections)
        self.out_channels = current_channels

    def __call__(self, x: chex.Array) -> chex.Array:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Output tensor of shape (batch, height, width, out_channels)
        """
        for conv, projection in zip(self.layers, self.projections):
            identity = x

            # Apply convolution
            out = conv(x)
            out = nnx.relu(out)

            # Apply projection if needed for residual
            if projection is not None:
                identity = projection(identity)

            # Residual connection
            x = out + identity

        return x
