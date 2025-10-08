from typing import List, Optional, Tuple
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
from agents.blocks import CNNBlock
import jax
import jax.numpy as jnp
import chex
import distrax
import envs.mytypes as env_types


class LbfFeatureExtractor(nnx.Module):
    """CNN-based feature extractor for LBF grid observations.

    Takes a 3-channel grid observation (agent layer, food layer, accessibility layer),
    normalizes it, then applies multiple CNN blocks with max pooling between them.
    Final output is global max pooled to 1x1xchannels.
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        cnn_blocks: List[Tuple[int, ...]],
    ):
        """Initialize CNN feature extractor.

        Args:
            key: Random key for initialization
            cnn_blocks: List of tuples, each tuple specifies channel sizes for a CNN block.
                       e.g., [(32, 64), (128, 256)] creates 2 blocks
        """
        keys = jax.random.split(key, len(cnn_blocks))

        # Build CNN blocks with max pooling
        blocks = []
        in_channels = 3  # GridObserver always produces 3 channels

        for key_block, channel_sizes in zip(keys, cnn_blocks):
            block = CNNBlock(
                key=key_block,
                in_channels=in_channels,
                channel_sizes=channel_sizes,
                kernel_size=3
            )
            blocks.append(block)
            in_channels = block.out_channels

        self.blocks = nnx.List(blocks)

        # Output dimension after global max pooling
        self.output_dim = in_channels

    def __call__(self, observations: chex.Array) -> chex.Array:
        """Forward pass through CNN feature extractor.

        Args:
            observations: Grid observations of shape (batch, 3, height, width) in NCHW format
                         Channel 0: agent levels (int)
                         Channel 1: food levels (int)
                         Channel 2: accessibility mask (0 or 1)

        Returns:
            Features of shape (batch, output_dim)
        """
        # Convert to float
        x = observations.astype(jnp.float32)

        # Normalize channels 0 and 1 (agent and food levels) by dividing by 6.0
        # Channel 2 (accessibility) is already binary (0/1), no normalization needed
        x = x.at[:, 0].divide(6.0)  # Agent layer
        x = x.at[:, 1].divide(6.0)  # Food layer
        # Channel 2 stays as is (0/1 binary mask)

        # Transpose from NCHW to NHWC format for Flax Conv
        # (batch, channels, height, width) -> (batch, height, width, channels)
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Apply CNN blocks with max pooling between them
        for block in self.blocks:
            # Apply CNN block
            x = block(x)

            # Max pool to reduce spatial dimensions by 2
            x = nnx.max_pool(
                x,
                window_shape=(2, 2),
                strides=(2, 2),
                padding='SAME'
            )

        # Global max pool to get 1x1xchannels
        # Pool over height and width dimensions (axes 1 and 2)
        x = jnp.max(x, axis=(1, 2))  # (batch, channels)

        return x


class LbfAgent(BaseAgent):
    """CNN-based agent specifically designed for the Level-Based Foraging environment."""

    def __init__(
        self,
        key: chex.PRNGKey,
        output_dim: int,
        cnn_blocks: List[Tuple[int, ...]] = [[64, 64], [128, 128]],
    ):
        """Initialize CNN agent for LBF.

        Args:
            key: Random key for initialization
            output_dim: Action space size (6 for LBF: noop, up, down, left, right, load)
            cnn_blocks: List of tuples specifying CNN block channels (default [[64, 64], [128, 128]])
        """

        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)

        # Separate feature extractors for policy and critic (no parameter sharing)
        self.policy_extractor = LbfFeatureExtractor(key1, cnn_blocks)
        self.critic_extractor = LbfFeatureExtractor(key2, cnn_blocks)

        # Policy and critic heads
        feature_dim = self.policy_extractor.output_dim
        self._policy_head = nnx.Linear(
            in_features=feature_dim,
            out_features=output_dim,
            rngs=rngs
        )
        self._critic_head = nnx.Linear(
            in_features=feature_dim,
            out_features=1,
            rngs=rngs
        )

        # Initialize modules
        layer_init(self, rngs.param())
        layer_init(self._policy_head, rngs.param(), std=0.01)

    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value."""
        features: chex.Array = self.critic_extractor(observations)
        return self._critic_head(features).squeeze(-1)

    def get_action(
        self,
        observations: env_types.Observation,
        key: chex.PRNGKey,
        action_masks: Optional[chex.Array] = None
    ) -> chex.Array:
        """Sample action from policy."""
        return self.get_action_distribution(observations, action_masks).sample(seed=key)

    def get_action_and_value(
        self,
        observations: env_types.Observation,
        key: chex.PRNGKey,
        action_masks: Optional[chex.Array] = None
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

    def get_action_distribution(
        self,
        observations: env_types.Observation,
        action_masks: Optional[chex.Array] = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network."""
        policy_features: chex.Array = self.policy_extractor(observations)
        logits: chex.Array = self._policy_head(policy_features)
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        return distrax.Categorical(logits=logits)
