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


class ConnectorFeatureExtractor(nnx.Module):
    """CNN-based feature extractor for grid observations.

    Takes a 2D grid with class indices, applies embedding, then multiple CNN blocks
    with max pooling between them. Final output is global max pooled to 1x1xchannels.
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        num_classes: int,
        embed_dim: int,
        cnn_blocks: List[Tuple[int, ...]],
    ):
        """Initialize CNN feature extractor.

        Args:
            key: Random key for initialization
            num_classes: Number of classes for embedding (num_agents * 3)
            embed_dim: Embedding dimension
            cnn_blocks: List of tuples, each tuple specifies channel sizes for a CNN block.
                       e.g., [(32, 64), (128, 256)] creates 2 blocks
        """
        keys = jax.random.split(key, len(cnn_blocks) + 1)
        key_embed = keys[0]
        keys_blocks = keys[1:]

        # Embedding layer for class indices
        self.embedding = nnx.Embed(
            num_embeddings=num_classes,
            features=embed_dim,
            rngs=nnx.Rngs(key_embed)
        )

        # Build CNN blocks with max pooling
        self.blocks = []
        in_channels = embed_dim

        for key_block, channel_sizes in zip(keys_blocks, cnn_blocks):
            block = CNNBlock(
                key=key_block,
                in_channels=in_channels,
                channel_sizes=channel_sizes,
                kernel_size=3
            )
            self.blocks.append(block)
            in_channels = block.out_channels

        # Output dimension after global max pooling
        self.output_dim = in_channels

    def __call__(self, observations: chex.Array) -> chex.Array:
        """Forward pass through CNN feature extractor.

        Args:
            observations: Grid observations of shape (batch, height, width) with class indices

        Returns:
            Features of shape (batch, output_dim)
        """
        # Ensure observations are integers for embedding
        obs_int = observations.astype(jnp.int32)

        # Apply embedding: (batch, height, width) -> (batch, height, width, embed_dim)
        x = self.embedding(obs_int)

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


class ConnectorAgent(BaseAgent):
    """CNN-based agent specifically designed for the Connector environment."""

    def __init__(
        self,
        key: chex.PRNGKey,
        num_classes: int,
        output_dim: int,
        embed_dim: int = 64,
        cnn_blocks: List[Tuple[int, ...]] = None,
    ):
        """Initialize CNN agent.

        Args:
            key: Random key for initialization
            num_classes: Number of classes for embedding (num_agents * 3)
            output_dim: Action space size
            embed_dim: Embedding dimension (default 64, should match first CNN channel)
            cnn_blocks: List of tuples specifying CNN block channels (default [(64, 64), (128, 128)])
        """
        if cnn_blocks is None:
            cnn_blocks = [(64, 64), (128, 128)]

        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)

        # Separate feature extractors for policy and critic (no parameter sharing)
        self.policy_extractor = ConnectorFeatureExtractor(
            key1, num_classes, embed_dim, cnn_blocks
        )
        self.critic_extractor = ConnectorFeatureExtractor(
            key2, num_classes, embed_dim, cnn_blocks
        )

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
