"""Unity ML-Agents environment creation."""

from omegaconf import DictConfig
from envs.mytypes import BaseEnv
from envs.wrappers.unity_subprocess_wrapper import UnitySubprocessWrapper
from envs.wrappers.unity_subprocess_vec_env import UnitySubprocessVecEnv


def create_unity_env(env_config: DictConfig, num_env: int = 1) -> BaseEnv:
    """
    Create Unity ML-Agents environment(s).

    Supports both single-instance and multi-instance modes:
    - Single instance: One Unity process with num_areas training areas
    - Multi-instance: Multiple Unity processes, each with num_areas training areas

    Args:
        env_config: Configuration with Unity-specific parameters:
            - file_name (str|None): Path to Unity executable (None for Editor mode)
            - base_port (int): Base port for communication (default: 5005)
            - time_scale (float): Unity time scale for faster training (default: 20.0)
            - seed (int): Random seed (default: 0)
            - no_graphics (bool): Disable graphics (default: True)
            - additional_args (list): Additional Unity command line args (default: None)
            - num_areas (int): Number of parallel training areas per instance (required)
        num_env: Total number of parallel environments desired

    Returns:
        UnitySubprocessWrapper (single instance) or UnitySubprocessVecEnv (multi-instance)

    Raises:
        ValueError: If num_areas not specified or num_env not divisible by num_areas

    Multi-Instance Architecture:
        num_instances = num_env // num_areas
        Each instance runs on unique port: base_port + worker_id
        Total parallel envs = num_instances * num_areas

    Example configs:
        # Single instance with 4 areas
        env_name: unity
        num_areas: 4
        # Call with num_env=4

        # Multi-instance: 2 instances × 4 areas = 8 total envs
        env_name: unity
        num_areas: 4
        # Call with num_env=8
    """
    # Get num_areas from config
    if "num_areas" not in env_config:
        raise ValueError(
            "num_areas must be specified in Unity environment config. "
            "This defines how many training areas each Unity instance has."
        )

    num_areas = env_config.num_areas

    # Validate num_env is divisible by num_areas
    if num_env % num_areas != 0:
        raise ValueError(
            f"num_env ({num_env}) must be divisible by num_areas ({num_areas}). "
            f"num_env should be a multiple of num_areas. "
            f"Valid values: {num_areas}, {num_areas*2}, {num_areas*3}, ..."
        )

    # Calculate number of Unity instances needed
    num_instances = num_env // num_areas

    if num_instances == 1:
        # Single instance mode
        return UnitySubprocessWrapper(
            env_config=env_config,
            num_areas=num_areas,
            worker_id=0,
            step_queue=None  # Sync mode
        )
    else:
        # Multi-instance mode
        return UnitySubprocessVecEnv(
            env_config=env_config,
            num_instances=num_instances,
            num_areas=num_areas
        )
