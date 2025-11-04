from typing import Tuple, Union, Literal
import jax
import jax.numpy as jnp
import chex
import distrax
from flax import nnx

from agents import BaseAgent
from agents.utils import layer_init
import envs.mytypes as env_types


ActionSpaceType = Literal["discrete", "multi_discrete", "continuous"]


class FeatureExtractor(nnx.Module):
    """Simple MLP-based feature extractor."""

    def __init__(self, key: chex.PRNGKey, input_dim: int, mlp_dim: int, num_hidden_layers: int = 1):
        rngs = nnx.Rngs(key)
        layers = [nnx.Linear(input_dim, mlp_dim, rngs=rngs), nnx.relu]

        for _ in range(num_hidden_layers - 1):
            layers += [nnx.Linear(mlp_dim, mlp_dim, rngs=rngs), nnx.relu]

        self.mlp = nnx.Sequential(*layers)

    def __call__(self, observations: chex.Array) -> chex.Array:
        flattened = observations.reshape(observations.shape[0], -1).astype(jnp.float32)
        return self.mlp(flattened)


class MLPAgent(BaseAgent):
    """Initialize MLPAgent with support for discrete, multi-discrete, and continuous actions.

        Args:
            key: JAX random key
            input_dim: Input observation dimension
            output_dim: Output action dimension(s).
                       - int: for Discrete action space (e.g., 5) or Continuous (e.g., 2)
                       - Tuple[int, ...]: for MultiDiscrete action space (e.g., (3, 2, 4))
            action_space_type: Type of action space - "discrete", "multi_discrete", or "continuous"
            mlp_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers
        """

    def __init__(
        self,
        key: chex.PRNGKey,
        input_dim: int,
        output_dim: Union[int, Tuple[int, ...]],
        action_space_type: ActionSpaceType = "discrete",
        mlp_dim: int = 64,
        num_hidden_layers: int = 3,
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)

        self.action_space_type = action_space_type
        self.output_dim = output_dim

        # Separate feature extractors for policy and critic
        self.policy_extractor = FeatureExtractor(key1, input_dim, mlp_dim, num_hidden_layers)
        self.critic_extractor = FeatureExtractor(key2, input_dim, mlp_dim, num_hidden_layers)

        # Policy heads
        if action_space_type == "multi_discrete":
            self._policy_heads = nnx.List([
                nnx.Linear(mlp_dim, n_actions, rngs=rngs)
                for n_actions in output_dim
            ])
        elif action_space_type == "continuous":
            assert isinstance(output_dim, int), f"Continuous actions require int output_dim, got {type(output_dim)}"
            self._policy_mean = nnx.Linear(mlp_dim, output_dim, rngs=rngs)
            self._policy_log_std = nnx.Param(jnp.zeros(output_dim))
        else:  # discrete
            self._policy_head = nnx.Linear(mlp_dim, output_dim, rngs=rngs)

        # Critic head
        self._critic_head = nnx.Linear(mlp_dim, 1, rngs=rngs)

        # Initialize layers
        layer_init(self, rngs.param())
        self._init_policy_heads(rngs)

    def _init_policy_heads(self, rngs: nnx.Rngs):
        """Initialize policy heads with small std."""
        if self.action_space_type == "multi_discrete":
            for head in self._policy_heads:
                layer_init(head, rngs.param(), std=0.01)
        elif self.action_space_type == "continuous":
            layer_init(self._policy_mean, rngs.param(), std=0.01)
        else:
            layer_init(self._policy_head, rngs.param(), std=0.01)

    # ===== Value functions =====

    @jax.jit
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value estimate."""
        return self._critic_head(self.critic_extractor(observations)).squeeze(-1)

    # ===== Policy functions =====

    @jax.jit
    def get_action_distribution(self, observations: env_types.Observation) -> distrax.Distribution:
        """Sample action and compute log probability and value.

        For continuous actions:
            - action_masks: not supported (ignored if provided)
            - returns actions of shape: (batch_size, action_dim)
        For multi-discrete actions:
            - action_masks shape: (batch_size, len(output_dim)) - must be all True (no masking supported)
            - returns actions of shape: (batch_size, len(output_dim))
        For discrete actions:
            - action_masks shape: (batch_size, n_actions)
            - returns actions of shape: (batch_size,)
        """
        features = self.policy_extractor(observations)

        if self.action_space_type == "multi_discrete":
            logits = jnp.stack([head(features) for head in self._policy_heads], axis=1)
            return distrax.Independent(distrax.Categorical(logits=logits), reinterpreted_batch_ndims=1)

        elif self.action_space_type == "continuous":
            mean = self._policy_mean(features)
            std = jnp.exp(self._policy_log_std)
            return distrax.Independent(distrax.Normal(mean, std), reinterpreted_batch_ndims=1)

        # Discrete
        logits = self._policy_head(features)
        return distrax.Categorical(logits=logits)

    @jax.jit
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey) -> chex.Array:
        """Sample an action from the current policy."""
        return self.get_action_distribution(observations).sample(seed=key)

    @jax.jit
    def get_action_and_value(
        self, observations: env_types.Observation, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Get action distribution from policy network.

        Returns:
            For continuous: MultivariateNormalDiag distribution
            For multi-discrete: Independent distribution wrapping Categorical distributions
            For discrete: Categorical distribution
        """
        dist = self.get_action_distribution(observations)
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        values = self.get_value(observations)
        return actions, log_probs, values
