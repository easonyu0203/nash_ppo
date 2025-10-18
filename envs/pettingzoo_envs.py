"""PettingZoo environment creators for nash_ppo."""

import warnings
# Suppress pkg_resources deprecation warning from pygame/pettingzoo
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

from omegaconf import DictConfig
from envs.wrappers import PettingZooWrapper, AddAgentIDWrapper
from supersuit.generic_wrappers import resize_v1, dtype_v0, normalize_obs_v0


def create_mpe_env(env_config: DictConfig, **kwargs):
    """Create a PettingZoo MPE (Multi-agent Particle Environment) wrapped for BaseEnv interface.

    MPE environments have vector observations and discrete actions.
    We add agent ID to observations so agents can distinguish themselves.

    Note: Environments with heterogeneous observation spaces (like simple_push_v3 where
    adversary and agent have different obs shapes) are automatically handled with padding.

    Args:
        env_config: Config with 'mpe_env_name' field (e.g., 'simple_push_v3', 'simple_v3')
                   and optional 'max_cycles' field (default: 100)
        **kwargs: Additional arguments to pass to the MPE environment

    Returns:
        Wrapped environment instance
    """
    from mpe2 import simple_push_v3, simple_v3, simple_tag_v3

    env_name = env_config.get("mpe_env_name")
    if env_name is None:
        raise ValueError("env_config must contain 'mpe_env_name' field for MPE environments")

    # Get max_cycles from config (default: 100)
    max_cycles = env_config.get("max_cycles", 100)

    # Map environment names to their constructors
    mpe_env_map = {
        "simple_push_v3": simple_push_v3.parallel_env,
        "simple_v3": simple_v3.parallel_env,
        "simple_tag_v3": simple_tag_v3.parallel_env,
    }

    if env_name not in mpe_env_map:
        raise ValueError(
            f"Unknown MPE environment: '{env_name}'. "
            f"Supported: {list(mpe_env_map.keys())}"
        )

    # Create the PettingZoo parallel environment
    pz_env = mpe_env_map[env_name](max_cycles=max_cycles, **kwargs)

    # Wrap for BaseEnv interface
    env = PettingZooWrapper(pz_env)

    # Add agent ID to observations (so agents know which agent they are)
    # For MPE (vector observations), this appends one-hot agent ID
    env = AddAgentIDWrapper(env, mode="vector")

    return env


def create_atari_env(env_config: DictConfig, **kwargs):
    """Create a PettingZoo Atari environment wrapped for BaseEnv interface.

    Atari environments have image observations (210x160x3) and discrete actions.
    We apply standard preprocessing: resize, normalize, and add agent ID channels.

    Args:
        env_config: Config with 'atari_env_name' field (e.g., 'boxing_v2', 'pong_v3', 'space_war_v2')
        **kwargs: Additional arguments to pass to the Atari environment

    Returns:
        Wrapped environment instance
    """
    from pettingzoo.atari import boxing_v2, pong_v3, space_war_v2

    env_name = env_config.get("atari_env_name")
    if env_name is None:
        raise ValueError("env_config must contain 'atari_env_name' field for Atari environments")

    # Map environment names to their constructors
    atari_env_map = {
        "boxing_v2": boxing_v2.parallel_env,
        "pong_v3": pong_v3.parallel_env,
        "space_war_v2": space_war_v2.parallel_env,
    }

    if env_name not in atari_env_map:
        raise ValueError(
            f"Unknown Atari environment: '{env_name}'. "
            f"Supported: {list(atari_env_map.keys())}"
        )

    # Create the PettingZoo parallel environment
    pz_env = atari_env_map[env_name](**kwargs)

    # Apply preprocessing wrappers from supersuit
    # 1. Resize to smaller image (84x84 is standard for Atari)
    target_size = env_config.get("image_size", 84)
    pz_env = resize_v1(pz_env, x_size=target_size, y_size=target_size)

    # 2. Convert to float32 and normalize to [0, 1]
    pz_env = dtype_v0(pz_env, dtype='float32')
    pz_env = normalize_obs_v0(pz_env, env_min=0.0, env_max=1.0)

    # Wrap for BaseEnv interface
    env = PettingZooWrapper(pz_env)

    # Add agent ID to observations (as additional image channels)
    env = AddAgentIDWrapper(env, mode="image")

    return env


__all__ = ["create_mpe_env", "create_atari_env"]
