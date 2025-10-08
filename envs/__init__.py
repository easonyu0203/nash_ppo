from omegaconf import DictConfig
from envs.wrappers import AutoResetWrapper, JumanjiWrapper
from envs.jumanji_envs import create_robot_warehouse, create_connector, create_lbf


# Registry mapping env_name to creator function
_ENV_REGISTRY = {
    "robot_warehouse": create_robot_warehouse,
    "connector": create_connector,
    "lbf": create_lbf,
    # Add more environments here as needed
    # "custom_env": create_custom_env,
}


def create_env(env_config: DictConfig, auto_reset: bool = True):
    """Create an environment instance based on the config.

    Args:
        env_config: The config object with `env_name` and environment-specific parameters
        auto_reset: If True, wrap environment with AutoResetWrapper (default: True)

    Returns:
        A wrapped environment instance

    Raises:
        ValueError: If the environment name is not recognized
    """
    env_name = env_config.env_name

    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment name: '{env_name}'. "
            f"Available environments: {list(_ENV_REGISTRY.keys())}"
        )

    creator_fn = _ENV_REGISTRY[env_name]

    env = creator_fn(env_config)

    # Optionally wrap with AutoResetWrapper
    if auto_reset:
        env = AutoResetWrapper(env)

    return env


__all__ = ["create_env", "AutoResetWrapper", "JumanjiWrapper"]
