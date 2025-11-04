from typing import Optional, Tuple, Union, Literal
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax
import jax.numpy as jnp
import chex
import distrax
import envs.mytypes as env_types

ActionSpaceType = Literal["discrete", "multi_discrete", "continuous"]

class FeatureExtractor(nnx.Module):

    def __init__(self, key: chex.PRNGKey, input_dim: int, mlp_dim: int, num_hidden_layers: int = 1):
        rngs = nnx.Rngs(key)

        # Build sequential layers
        layers = []
        # First layer: input_dim -> mlp_dim
        layers.extend([
            nnx.Linear(in_features=input_dim, out_features=mlp_dim, rngs=rngs),
            nnx.relu
        ])
        # Hidden layers: mlp_dim -> mlp_dim
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nnx.Linear(in_features=mlp_dim, out_features=mlp_dim, rngs=rngs),
                nnx.relu
            ])

        self.mlp = nnx.Sequential(*layers)
        
    def __call__(self, observations: chex.Array) -> chex.Array:
        # Flatten input
        flattened = observations.reshape(observations.shape[0], -1).astype(jnp.float32)
        result: chex.Array = self.mlp(flattened)
        return result

class MLPAgent(BaseAgent):

    def __init__(
        self,
        key: chex.PRNGKey,
        input_dim: int,
        output_dim: Union[int, Tuple[int, ...]],
        action_space_type: ActionSpaceType = "discrete",
        mlp_dim: int = 64,
        num_hidden_layers: int = 3
    ):
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
        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)

        # Store action space type and dimensions
        self.action_space_type = action_space_type
        self.output_dim = output_dim

        # Separate feature extractors for policy and critic (no parameter sharing)
        self.policy_extractor = FeatureExtractor(key1, input_dim, mlp_dim, num_hidden_layers)
        self.critic_extractor = FeatureExtractor(key2, input_dim, mlp_dim, num_hidden_layers)

        # Policy and critic heads
        if action_space_type == "multi_discrete":
            # Create separate head for each action dimension
            # Each head outputs logits for its respective action space
            self._policy_heads = nnx.List([
                nnx.Linear(in_features=mlp_dim, out_features=n_actions, rngs=rngs)
                for n_actions in output_dim
            ])
        elif action_space_type == "continuous":
            # For continuous actions: output mean values
            # output_dim should be an int representing the action dimension
            assert isinstance(output_dim, int), f"For continuous actions, output_dim must be int, got {type(output_dim)}"
            self._policy_mean = nnx.Linear(in_features=mlp_dim, out_features=output_dim, rngs=rngs)
            # State-independent log_std as a learnable parameter
            self._policy_log_std = nnx.Param(jnp.zeros(output_dim))
        else:  # discrete
            # Single head for discrete actions
            self._policy_head = nnx.Linear(in_features=mlp_dim, out_features=output_dim, rngs=rngs)

        self._critic_head = nnx.Linear(in_features=mlp_dim, out_features=1, rngs=rngs)

        # Initialize modules
        layer_init(self, rngs.param())
        if action_space_type == "multi_discrete":
            for head in self._policy_heads:
                layer_init(head, rngs.param(), std=0.01)
        elif action_space_type == "continuous":
            layer_init(self._policy_mean, rngs.param(), std=0.01)
        else:
            layer_init(self._policy_head, rngs.param(), std=0.01)

    @jax.jit
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value."""
        features: chex.Array = self.critic_extractor(observations)
        return self._critic_head(features).squeeze(-1)

    @jax.jit
    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None) -> chex.Array:
        """Sample action from policy."""
        return self.get_action_distribution(observations, action_masks).sample(seed=key)
    
    @jax.jit
    def get_action_and_value(
            self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None
        ) -> Tuple[chex.Array, chex.Array, chex.Array]:
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
        policy_features: chex.Array = self.policy_extractor(observations)

        if self.action_space_type == "multi_discrete":
            # Multi-discrete: action masking not supported (action_masks ignored if provided)
            # Get logits for each action dimension
            logits_list = [head(policy_features) for head in self._policy_heads]
            # Stack to shape: (batch_size, n_dims, n_actions_per_dim)
            logits = jnp.stack(logits_list, axis=1)

            # Create Independent distribution over Categorical distributions
            categoricals = distrax.Categorical(logits=logits)
            dist = distrax.Independent(categoricals, reinterpreted_batch_ndims=1)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
        elif self.action_space_type == "continuous":
            # Continuous: action masking not supported (action_masks ignored if provided)
            # Get mean from policy network
            mean: chex.Array = self._policy_mean(policy_features)
            # Get std from state-independent log_std parameter
            std = jnp.exp(self._policy_log_std)

            # Create independent normal distribution for each action dimension
            # Using Normal + Independent instead of MultivariateNormalDiag to avoid tracer leaks
            normal_dist = distrax.Normal(loc=mean, scale=std)
            dist = distrax.Independent(normal_dist, reinterpreted_batch_ndims=1)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
        else:  # discrete
            # Discrete action space (original behavior)
            logits: chex.Array = self._policy_head(policy_features)
            if action_masks is not None:
                logits = jnp.where(action_masks, logits, -jnp.inf)

            dist = distrax.Categorical(logits=logits)
            actions, log_probs = dist.sample_and_log_prob(seed=key)

        critic_features: chex.Array = self.critic_extractor(observations)
        values = self._critic_head(critic_features).squeeze(-1)

        return actions, log_probs, values

    @jax.jit
    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: Optional[chex.Array] = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network.

        Returns:
            For continuous: MultivariateNormalDiag distribution
            For multi-discrete: Independent distribution wrapping Categorical distributions
            For discrete: Categorical distribution
        """
        policy_features: chex.Array = self.policy_extractor(observations)

        if self.action_space_type == "multi_discrete":
            # Multi-discrete: action masking not supported (action_masks ignored if provided)
            # Get logits for each action dimension
            logits_list = [head(policy_features) for head in self._policy_heads]
            # Stack to shape: (batch_size, n_dims, n_actions_per_dim)
            logits = jnp.stack(logits_list, axis=1)

            # Create Independent distribution over Categorical distributions
            categoricals = distrax.Categorical(logits=logits)
            return distrax.Independent(categoricals, reinterpreted_batch_ndims=1)
        elif self.action_space_type == "continuous":
            # Continuous: action masking not supported (action_masks ignored if provided)
            # Get mean from policy network
            mean: chex.Array = self._policy_mean(policy_features)
            # Get std from state-independent log_std parameter
            std = jnp.exp(self._policy_log_std)

            # Create independent normal distribution for each action dimension
            # Using Normal + Independent instead of MultivariateNormalDiag to avoid tracer leaks
            normal_dist = distrax.Normal(loc=mean, scale=std)
            return distrax.Independent(normal_dist, reinterpreted_batch_ndims=1)
        else:  # discrete
            # Discrete action space (original behavior)
            logits: chex.Array = self._policy_head(policy_features)
            if action_masks is not None:
                logits = jnp.where(action_masks, logits, -jnp.inf)
            return distrax.Categorical(logits=logits)