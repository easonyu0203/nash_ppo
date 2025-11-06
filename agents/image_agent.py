"""CNN-based agent for image observations."""

from typing import Tuple, Sequence
import jax
import chex

from agents.base import ActorCriticAgent, ActionSpaceType
from agents.networks.feature_extractors import CNNFeatureExtractor


class ImageAgent(ActorCriticAgent):
    """CNN-based agent for image observations.

    This agent uses CNN feature extractors for both policy and value networks,
    making it suitable for image-based observations (e.g., Atari games, visual tasks).

    Currently supports discrete action spaces only.

    Args:
        key: JAX random key
        input_shape: Shape of input images (H, W, C) - channels last
        output_dim: Number of discrete actions
        cnn_channels: Number of channels for each CNN layer (default: 4 layers)
        kernel_size: Kernel size for all conv layers (default: 3x3)
        pool_size: Max pooling factor (default: 2, reduces H,W by 2 after each layer)
        normalize: Whether to normalize input images (default: False)
        normalization_factor: Value to divide by for normalization (default: 255.0)

    Example:
        >>> # Atari-style agent
        >>> agent = ImageAgent(
        ...     key,
        ...     input_shape=(210, 160, 3),
        ...     output_dim=4,
        ...     normalize=True
        ... )
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        input_shape: Tuple[int, ...],
        output_dim: int,
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        pool_size: int = 2,
        normalize: bool = False,
        normalization_factor: float = 255.0,
    ):
        """Initialize image-based agent."""
        # Split keys for policy and critic networks
        key_policy, key_critic = jax.random.split(key)

        # Create separate CNN feature extractors for policy and critic
        policy_extractor = CNNFeatureExtractor(
            key_policy,
            input_shape,
            cnn_channels,
            kernel_size,
            pool_size,
            normalize,
            normalization_factor,
        )
        critic_extractor = CNNFeatureExtractor(
            key_critic,
            input_shape,
            cnn_channels,
            kernel_size,
            pool_size,
            normalize,
            normalization_factor,
        )

        # Initialize parent actor-critic agent
        # Note: Currently only supports discrete actions for image-based agents
        super().__init__(
            key=key,
            policy_extractor=policy_extractor,
            critic_extractor=critic_extractor,
            action_dim=output_dim,
            action_space_type=ActionSpaceType.DISCRETE,
        )
