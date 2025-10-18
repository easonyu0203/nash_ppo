"""Gymnasium environment creators for nash_ppo."""

import gymnasium as gym
from omegaconf import DictConfig
from envs.wrappers import GymnasiumWrapper


def create_gymnasium_env(env_config: DictConfig, **kwargs):
    """Create a Gymnasium environment wrapped for BaseEnv interface.

    Args:
        env_config: Config with 'gym_env_id' field specifying the Gymnasium environment ID
        **kwargs: Additional arguments to pass to gym.make (e.g., render_mode="human")

    Returns:
        GymnasiumWrapper instance
    """
    env_id = env_config.get("gym_env_id")
    if env_id is None:
        raise ValueError("env_config must contain 'gym_env_id' field for gymnasium environments")

    # Create the gymnasium environment with any additional kwargs
    gym_env = gym.make(env_id, **kwargs)

    # Wrap it for BaseEnv interface
    return GymnasiumWrapper(gym_env)


__all__ = ["create_gymnasium_env"]
