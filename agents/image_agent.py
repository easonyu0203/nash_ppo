from functools import partial
from typing import Optional, Tuple, Sequence
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax
import jax.numpy as jnp
import chex
import distrax
import envs.mytypes as env_types


class CNNFeatureExtractor(nnx.Module):

    def __init__(
        self,
        key: chex.PRNGKey,
        input_shape: Tuple[int, ...],  # (H, W, C) - channels last
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        pool_size: int = 2,
        normalize: bool = False,
        normalization_factor: float = 255.0
    ):
        """
        CNN-based feature extractor for image observations.

        Architecture:
        - Each CNN layer: Conv(3x3, SAME) -> ReLU -> MaxPool(4x4, stride 4)
        - Final: Global MaxPool over spatial dimensions -> feature vector

        Args:
            key: JAX random key
            input_shape: Shape of input images (H, W, C) - channels last
            cnn_channels: Number of channels for each CNN layer (default 4 layers for Atari)
            kernel_size: Kernel size for all conv layers (default: 3)
            pool_size: Max pooling size and stride (default: 2, reduces H,W by factor of 2)
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
        self.num_output_features = in_channels

    def __call__(self, observations: chex.Array) -> chex.Array:
        """
        Process image observations through CNN.

        Args:
            observations: Image tensor of shape (batch_size, H, W, C)

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


class ImageAgent(BaseAgent):

    def __init__(
        self,
        key: chex.PRNGKey,
        input_shape: Tuple[int, ...],  # Image shape (H, W, C) - channels last
        output_dim: int,  # Number of actions
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        pool_size: int = 2,
        normalize: bool = False,
        normalization_factor: float = 255.0
    ):
        """
        Image-based agent using CNN for feature extraction.
        Designed for Atari-style games with image observations (e.g., 210x160x3).

        Args:
            key: JAX random key
            input_shape: Shape of input images (H, W, C) - channels last (e.g., (210, 160, 3))
            output_dim: Number of actions (output dimension)
            cnn_channels: Number of channels for each CNN layer (default: 4 layers)
            kernel_size: Kernel size for all conv layers (default: 3x3)
            pool_size: Max pooling factor (default: 2, reduces H,W by 2 after each layer)
            normalize: Whether to normalize input images (default: True)
            normalization_factor: Value to divide by for normalization (default: 255.0)
        """
        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)

        # Separate feature extractors for policy and critic (no parameter sharing)
        self.policy_extractor = CNNFeatureExtractor(
            key1, input_shape, cnn_channels, kernel_size, pool_size,
            normalize, normalization_factor
        )
        self.critic_extractor = CNNFeatureExtractor(
            key2, input_shape, cnn_channels, kernel_size, pool_size,
            normalize, normalization_factor
        )

        # Policy and critic heads (input is feature vector from global max pool)
        feature_dim = self.policy_extractor.num_output_features
        self._policy_head = nnx.Linear(in_features=feature_dim, out_features=output_dim, rngs=rngs)
        self._critic_head = nnx.Linear(in_features=feature_dim, out_features=1, rngs=rngs)

        # Initialize modules
        layer_init(self, rngs.param())
        layer_init(self._policy_head, rngs.param(), std=0.01)

    @partial(jax.jit, static_argnames=('self', ))
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value."""
        features: chex.Array = self.critic_extractor(observations)
        return self._critic_head(features).squeeze(-1)

    @partial(jax.jit, static_argnames=('self', ))
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None) -> chex.Array:
        """Sample action from policy."""
        return self.get_action_distribution(observations, action_masks).sample(seed=key)

    @partial(jax.jit, static_argnames=('self', ))
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Sample action and compute log probability and value."""
        policy_features: chex.Array = self.policy_extractor(observations)
        logits: chex.Array = self._policy_head(policy_features)
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)

        dist = distrax.Categorical(logits=logits)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        critic_features: chex.Array = self.critic_extractor(observations)
        values = self._critic_head(critic_features).squeeze(-1)

        return actions, log_probs, values

    @partial(jax.jit, static_argnames=('self', ))
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: Optional[chex.Array] = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network."""
        policy_features: chex.Array = self.policy_extractor(observations)
        logits: chex.Array = self._policy_head(policy_features)
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        return distrax.Categorical(logits=logits)