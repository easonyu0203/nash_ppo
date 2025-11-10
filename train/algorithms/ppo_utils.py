"""Shared utilities for PPO training."""

from typing import Union, Tuple
import jax.numpy as jnp
import chex

from agents.base.types import ActionSpaceType


def get_action_norm_factor(
    action_dim: Union[int, Tuple[int, ...]],
    action_space_type: ActionSpaceType,
    normalize_logprob: bool
) -> chex.Numeric:
    """Compute normalization factor for log probabilities based on action space.

    Args:
        action_dim: Action dimension(s) from agent
            - int for discrete/continuous
            - Tuple[int, ...] for multi-discrete
        action_space_type: Type of action space from agent
        normalize_logprob: Whether to normalize log probs by action dimensions

    Returns:
        Normalization factor:
        - 1.0 if not normalizing
        - action_dim for discrete/continuous
        - sum of action_dims for multi-discrete
    """
    if not normalize_logprob:
        return jnp.float32(1.0)

    # Compute number of action dimensions based on action space type
    if action_space_type == ActionSpaceType.DISCRETE:
        # Discrete: single action dimension
        n_action_dims = 1
    elif action_space_type == ActionSpaceType.CONTINUOUS:
        # Continuous: action_dim is the number of continuous dimensions
        n_action_dims = action_dim
    elif action_space_type == ActionSpaceType.MULTI_DISCRETE:
        # Multi-discrete: sum all discrete dimensions
        n_action_dims = sum(action_dim)
    else:
        raise ValueError(f"Unknown action space type: {action_space_type}")

    return jnp.float32(n_action_dims)
