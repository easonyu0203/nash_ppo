"""Unity ML-Agents environment creation."""

from omegaconf import DictConfig
from envs.mytypes import BaseEnv
from envs.wrappers.unity_subprocess_wrapper import UnitySubprocessWrapper


def create_unity_env(env_config: DictConfig, num_areas=1) -> BaseEnv:
    """
    Create Unity ML-Agents environment.

    Args:
        env_config: Configuration with Unity-specific parameters:
            - file_name (str|None): Path to Unity executable (None for Editor mode)
            - base_port (int): Base port for communication (default: 5004)
            - time_scale (float): Unity time scale for faster training (default: 20.0)
            - seed (int): Random seed (default: 0)
            - worker_id (int): Worker ID for parallel instances (default: 0)
            - no_graphics (bool): Disable graphics (default: True)
            - additional_args (list): Additional Unity command line args (default: None)
        - num_areas (int): Number of parallel training areas (default: 1)

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

    return UnitySubprocessWrapper(env_config, num_areas=num_areas)
