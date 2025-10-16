from omegaconf import DictConfig
import gymnasium as gym
from envs.wrappers import AutoResetWrapper, GymnasiumWrapper, DummyVecEnv


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


# Registry mapping env_name to creator function
_ENV_REGISTRY = {
    "gymnasium": create_gymnasium_env,
    # Add more environments here as needed
    # "custom_env": create_custom_env,
}


def create_env(env_config: DictConfig, auto_reset: bool = True, num_env: int = 1, **kwargs):
    """Create an environment instance based on the config.

    Args:
        env_config: The config object with `env_name` and environment-specific parameters
        auto_reset: If True, wrap environment with AutoResetWrapper (default: True)
        num_env: Number of parallel environments to create (default: 1)
        **kwargs: Additional keyword arguments to pass to the environment creator
                 (e.g., render_mode="human" for gymnasium environments)

    Returns:
        A wrapped environment instance (or DummyVecEnv if num_env > 1)

    Raises:
        ValueError: If the environment name is not recognized or num_env < 1
    """
    if num_env < 1:
        raise ValueError(f"num_env must be >= 1, got {num_env}")

    env_name = env_config.env_name

    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment name: '{env_name}'. "
            f"Available environments: {list(_ENV_REGISTRY.keys())}"
        )

    creator_fn = _ENV_REGISTRY[env_name]

    def make_env():
        """Helper function to create a single environment instance."""
        env = creator_fn(env_config, **kwargs)
        if auto_reset:
            env = AutoResetWrapper(env)
        return env

    # Create single env or vectorized env
    if num_env == 1:
        return make_env()
    else:
        # Create list of env creation functions
        env_fns = [make_env for _ in range(num_env)]
        return DummyVecEnv(env_fns)


__all__ = ["create_env", "AutoResetWrapper", "GymnasiumWrapper", "DummyVecEnv"]
