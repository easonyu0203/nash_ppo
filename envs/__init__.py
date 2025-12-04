from omegaconf import DictConfig
from envs.wrappers import AutoResetWrapper, GymnasiumWrapper, DummyVecEnv
from envs.gymnasium_envs import create_gymnasium_env
from envs.unity_envs import create_unity_env
from envs.squid_envs import create_squid_env

# Registry mapping env_name to creator function
_ENV_REGISTRY = {
    "gymnasium": create_gymnasium_env,
    "unity": create_unity_env,
    "squid": create_squid_env,
    # Add more environments here as needed
}

# Optional pettingzoo support
try:
    from envs.pettingzoo_envs import create_mpe_env, create_atari_env
    _ENV_REGISTRY["mpe"] = create_mpe_env
    _ENV_REGISTRY["atari"] = create_atari_env
except ImportError:
    pass  # pettingzoo not installed

def create_env(env_config: DictConfig, auto_reset: bool = True, num_env: int = 1, **kwargs):
    """Create an environment instance based on the config.

    Args:
        env_config: The config object with `env_name` and environment-specific parameters
        auto_reset: If True, wrap environment with AutoResetWrapper (default: True)
                   Note: Ignored for Unity environments (they auto-reset internally)
        num_env: Number of parallel environments to create (default: 1)
                Note: For Unity environments, this sets num_areas instead of vectorization
        **kwargs: Additional keyword arguments to pass to the environment creator
                 (e.g., render_mode="human" for gymnasium environments)

    Returns:
        A wrapped environment instance (or DummyVecEnv if num_env > 1)
        For Unity environments, returns a single UnityEnvWrapper instance

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

    # Unity environments handle parallelization and auto-reset internally
    if env_name == "unity":
        # Use num_env to set num_areas for Unity's internal parallelization
        # Unity environments auto-reset and parallelize internally
        return create_unity_env(env_config, num_env=num_env)

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
