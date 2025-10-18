from omegaconf import DictConfig
from envs.wrappers import AutoResetWrapper, GymnasiumWrapper, DummyVecEnv
from envs.gymnasium_envs import create_gymnasium_env
from envs.pettingzoo_envs import create_mpe_env, create_atari_env


# Registry mapping env_name to creator function
_ENV_REGISTRY = {
    "gymnasium": create_gymnasium_env,
    "mpe": create_mpe_env,
    "atari": create_atari_env,
    # Add more environments here as needed
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
