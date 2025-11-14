"""Tag agent for Unity Tag environment with ray sensor + velocity observations."""

from typing import Dict, Tuple, Union
import jax
import jax.numpy as jnp
import chex
from flax import nnx

from agents.base import StatefulActorCriticAgent, ActionSpaceType
from agents.networks.feature_extractors import LSTMFeatureExtractor
from agents.networks.feature_extractors.base import FeatureExtractor


class TagFeatureExtractor(FeatureExtractor):
    """Feature extractor for Unity Tag environment with ray sensor + auxiliary features.

    Unity Tag environment has Dict observation space:
    - 'obs_0': Ray sensor (85,) = 17 rays × 5 attributes
    - 'obs_1': Auxiliary features (3,) - e.g., velocity, orientation, etc.

    Architecture:
        1. Reshape ray sensor: (85,) -> (17, 5)
        2. Broadcast auxiliary features to all rays: (3,) -> (17, 3)
        3. Concatenate: (17, 8)
        4. 1D convolutions with max pooling: 8 -> 64 -> 128 -> 256
        5. Global max pooling: -> (256,)

    Args:
        key: JAX random key
        num_rays: Number of rays in ray sensor (default: 17)
        ray_attrs: Number of attributes per ray (default: 5)
        aux_dim: Dimension of auxiliary features (velocity, orientation, etc.) (default: 3)
        feature_channels: Tuple of output channels for conv layers (default: (64, 128, 256))
        kernel_size: Kernel size for 1D convolutions (default: 3)
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        num_rays: int = 17,
        ray_attrs: int = 5,
        aux_dim: int = 3,
        feature_channels: Tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 3,
    ):
        """Initialize Tag feature extractor."""
        self.num_rays = num_rays
        self.ray_attrs = ray_attrs
        self.aux_dim = aux_dim
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size

        # Input dimension after concatenating rays and auxiliary features
        input_channels = ray_attrs + aux_dim

        # Build 1D conv layers
        rngs = nnx.Rngs(key)
        conv_layers = []

        in_channels = input_channels
        for out_channels in feature_channels:
            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(kernel_size,),
                padding='SAME',
                rngs=rngs
            )
            conv_layers.append(conv)
            in_channels = out_channels

        self.conv_layers = nnx.List(conv_layers)
        self._output_dim = feature_channels[-1]

    def __call__(self, observations: Dict[str, chex.Array]) -> chex.Array:
        """Extract features from ray sensor + auxiliary observations.

        Args:
            observations: Dict with keys:
                - 'obs_0': Ray sensor (batch_size, 85)
                - 'obs_1': Auxiliary features (batch_size, 3)

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        batch_size = observations['obs_0'].shape[0]

        # 1. Reshape ray sensor: (batch, 85) -> (batch, 17, 5)
        rays = observations['obs_0'].reshape(batch_size, self.num_rays, self.ray_attrs)

        # 2. Broadcast auxiliary features to all rays: (batch, 3) -> (batch, 17, 3)
        aux_features = observations['obs_1']
        aux_broadcast = jnp.tile(
            aux_features[:, jnp.newaxis, :],  # (batch, 1, aux_dim)
            (1, self.num_rays, 1)             # -> (batch, num_rays, aux_dim)
        )

        # 3. Concatenate: (batch, 17, 5) + (batch, 17, 3) -> (batch, 17, 8)
        x = jnp.concatenate([rays, aux_broadcast], axis=-1)

        # 4. Apply conv layers with max pooling
        for conv in self.conv_layers:
            x = conv(x)       # 1D Convolution
            x = nnx.relu(x)   # Activation

            # Max pool stride 2: (batch, length, channels) -> (batch, length//2, channels)
            x = jax.lax.reduce_window(
                x,
                -jnp.inf,
                jax.lax.max,
                window_dimensions=(1, 2, 1),  # Pool over spatial dimension
                window_strides=(1, 2, 1),     # Stride 2
                padding='VALID'
            )

        # 5. Global max pooling: (batch, remaining_length, channels) -> (batch, channels)
        features = jnp.max(x, axis=1)

        return features

    @property
    def output_dim(self) -> int:
        """Output feature dimension."""
        return self._output_dim


class TagAgent(StatefulActorCriticAgent):
    """LSTM agent for Unity Tag environment with ray sensor observations.

    Architecture:
        Dict obs -> TagFeatureExtractor -> LSTM -> policy/value

    The Tag environment has:
    - Observation: Dict('obs_0': (85,), 'obs_1': (3,))
    - Action: MultiDiscrete([3, 3])

    Args:
        key: JAX random key
        num_rays: Number of rays in ray sensor (default: 17)
        ray_attrs: Number of attributes per ray (default: 5)
        aux_dim: Dimension of auxiliary features (velocity, orientation, etc.) (default: 3)
        output_dim: Output action dimension (default: (3, 3) for MultiDiscrete)
        action_space_type: Type of action space (default: "multi_discrete")
        feature_channels: Conv layer channels (default: (64, 128, 256))
        lstm_dim: LSTM hidden dimension (default: 128)
        num_lstm_layers: Number of stacked LSTM layers (default: 2)
        kernel_size: Conv kernel size (default: 3)

    Example:
        >>> agent = TagAgent(key)
        >>> carry = agent.initialize_carry(batch_size=4)
        >>> actions, log_probs, values, new_carry = agent.get_action_and_value(obs_dict, carry, key)
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        num_rays: int = 17,
        ray_attrs: int = 5,
        aux_dim: int = 3,
        output_dim: Union[int, Tuple[int, ...]] = (3, 3),
        action_space_type: Union[ActionSpaceType, str] = ActionSpaceType.MULTI_DISCRETE,
        feature_channels: Tuple[int, ...] = (64, 128, 256),
        lstm_dim: int = 128,
        num_lstm_layers: int = 2,
        kernel_size: int = 3,
    ):
        """Initialize Tag agent."""
        key_policy_base, key_critic_base, key_policy_lstm, key_critic_lstm = jax.random.split(key, 4)

        # Create Tag feature extractors for policy and critic
        policy_base = TagFeatureExtractor(
            key_policy_base,
            num_rays=num_rays,
            ray_attrs=ray_attrs,
            aux_dim=aux_dim,
            feature_channels=feature_channels,
            kernel_size=kernel_size
        )
        critic_base = TagFeatureExtractor(
            key_critic_base,
            num_rays=num_rays,
            ray_attrs=ray_attrs,
            aux_dim=aux_dim,
            feature_channels=feature_channels,
            kernel_size=kernel_size
        )

        # Wrap with LSTM
        policy_extractor = LSTMFeatureExtractor(
            key_policy_lstm, policy_base, lstm_dim, num_lstm_layers
        )
        critic_extractor = LSTMFeatureExtractor(
            key_critic_lstm, critic_base, lstm_dim, num_lstm_layers
        )

        # Initialize parent
        super().__init__(
            key=key,
            policy_extractor=policy_extractor,
            critic_extractor=critic_extractor,
            action_dim=output_dim,
            action_space_type=action_space_type,
        )
