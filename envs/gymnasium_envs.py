"""Gymnasium environment creators for nash_ppo."""

import gymnasium as gym
from omegaconf import DictConfig
from envs.wrappers import GymnasiumWrapper


def create_gymnasium_env(env_config: DictConfig, **kwargs):
    """Create a Gymnasium environment wrapped for BaseEnv interface.

    Args:
        env_config: Config with 'gym_env_id' field specifying the Gymnasium environment ID
                   Optional 'observable_dims' field to create POMDP (list of dimension indices)
        **kwargs: Additional arguments to pass to gym.make (e.g., render_mode="human")

    Returns:
        GymnasiumWrapper instance (optionally wrapped with PartialObservabilityWrapper)
    """
    env_id = env_config.get("gym_env_id")
    if env_id is None:
        raise ValueError("env_config must contain 'gym_env_id' field for gymnasium environments")

    # Create the gymnasium environment with any additional kwargs
    gym_env = gym.make(env_id, **kwargs)

    # Wrap it for BaseEnv interface
    wrapped_env = GymnasiumWrapper(gym_env)

    # Apply POMDP wrapper if observable_dims is specified
    if "observable_dims" in env_config:
        from envs.wrappers.partial_observability_wrapper import PartialObservabilityWrapper
        wrapped_env = PartialObservabilityWrapper(
            wrapped_env,
            observable_dims=list(env_config.observable_dims)
        )

    return wrapped_env


__all__ = ["create_gymnasium_env"]
