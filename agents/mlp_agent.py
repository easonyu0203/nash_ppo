from typing import Optional, Tuple
from flax import nnx
from agents import BaseAgent
from agents.utils import layer_init
import jax
import jax.numpy as jnp
import chex
import distrax
import envs.mytypes as env_types

class FeatureExtractor(nnx.Module):

    def __init__(self, key: chex.PRNGKey, input_dim: int, mlp_dim: int):
        rngs = nnx.Rngs(key)
        self.mlp = nnx.Sequential(
            nnx.Linear(in_features=input_dim, out_features=mlp_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=mlp_dim, out_features=mlp_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=mlp_dim, out_features=mlp_dim, rngs=rngs),
            nnx.relu
        )
        
    def __call__(self, observations: chex.Array) -> chex.Array:
        # Flatten input
        flattened = observations.reshape(observations.shape[0], -1).astype(jnp.float32)
        result: chex.Array = self.mlp(flattened)
        return result

class MLPAgent(BaseAgent):

    def __init__(self, key: chex.PRNGKey, input_dim: int, output_dim: int, mlp_dim: int = 64):
        key1, key2, key3 = jax.random.split(key, 3)
        rngs = nnx.Rngs(key3)

        # Separate feature extractors for policy and critic (no parameter sharing)
        self.policy_extractor = FeatureExtractor(key1, input_dim, mlp_dim)
        self.critic_extractor = FeatureExtractor(key2, input_dim, mlp_dim)

        # Policy and critic heads
        self._policy_head = nnx.Linear(in_features=mlp_dim, out_features=output_dim, rngs=rngs)
        self._critic_head = nnx.Linear(in_features=mlp_dim, out_features=1, rngs=rngs)

        # Initialize modules
        layer_init(self, rngs.param())
        layer_init(self._policy_head, rngs.param(), std=0.01)
        
    def get_value(self, observations: env_types.Observation) -> chex.Array:
        """Compute state value."""
        features: chex.Array = self.critic_extractor(observations)
        return self._critic_head(features).squeeze(-1)

    def get_action(self, observations: env_types.Observation, key: chex.PRNGKey, action_masks: Optional[chex.Array] = None) -> chex.Array:
        """Sample action from policy."""
        return self.get_action_distribution(observations, action_masks).sample(seed=key)
    
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

    def get_action_distribution(
        self, observations: env_types.Observation, action_masks: Optional[chex.Array] = None
    ) -> distrax.Distribution:
        """Get action distribution from policy network."""
        policy_features: chex.Array = self.policy_extractor(observations)
        logits: chex.Array = self._policy_head(policy_features)
        if action_masks is not None:
            logits = jnp.where(action_masks, logits, -jnp.inf)
        return distrax.Categorical(logits=logits)