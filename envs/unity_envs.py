"""Unity ML-Agents environment creation."""

from omegaconf import DictConfig
from envs.mytypes import BaseEnv
from envs.wrappers.unity_env_wrapper import UnityEnvWrapper


def create_unity_env(env_config: DictConfig) -> BaseEnv:
    """
    Create Unity ML-Agents environment.

    Args:
        env_config: Configuration with Unity-specific parameters:
            - file_name (str|None): Path to Unity executable (None for Editor mode)
            - num_areas (int): Number of parallel training areas (default: 1)
            - base_port (int): Base port for communication (default: 5004)
            - time_scale (float): Unity time scale for faster training (default: 20.0)
            - seed (int): Random seed (default: 0)
            - worker_id (int): Worker ID for parallel instances (default: 0)
            - no_graphics (bool): Disable graphics (default: True)
            - additional_args (list): Additional Unity command line args (default: None)

    Returns:
        UnityEnvWrapper instance

    Note:
        Unlike other environments, Unity uses num_areas for parallelization
        (via TrainingAreaReplicator) instead of multiple process instances.
        The num_env parameter in create_env() sets num_areas automatically.

    Example config:
        env_name: unity
        file_name: null  # Editor mode
        num_areas: 8
        time_scale: 20.0
        no_graphics: true
    """

    # Extract Unity-specific parameters from env_config
    file_name = env_config.get("file_name", None)
    num_areas = env_config.get("num_areas", 1)
    base_port = env_config.get("base_port", 5004)
    time_scale = env_config.get("time_scale", 20.0)
    seed = env_config.get("seed", 0)
    worker_id = env_config.get("worker_id", 0)
    no_graphics = env_config.get("no_graphics", True)
    additional_args = env_config.get("additional_args", None)

    # Create Unity environment wrapper
    env = UnityEnvWrapper(
        file_name=file_name,
        num_areas=num_areas,
        base_port=base_port,
        time_scale=time_scale,
        seed=seed,
        worker_id=worker_id,
        no_graphics=no_graphics,
        additional_args=additional_args,
    )

    return env
