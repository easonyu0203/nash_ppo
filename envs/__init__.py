import envs.mytypes as env_types
from envs.tictactoe import TicTacToe
from enum import Enum
from typing import Union
from envs.wrappers import AutoResetWrapper


class RegisteredEnv(Enum):
    TIC_TAC_TOE = "tic_tac_toe"


def create_env(env_name: Union[RegisteredEnv, str], auto_reset: bool = True) -> env_types.BaseEnv:
    """Create an environment instance based on the registered environment name.
    
    Args:
        env_name: The registered environment type to create (enum or string)
        auto_reset: Whether to wrap the environment with AutoResetWrapper (default: True)
        
    Returns:
        An instance of the specified environment type, optionally wrapped with AutoResetWrapper
        
    Raises:
        ValueError: If the environment name is not recognized
    """
    # Convert string to enum if needed
    if isinstance(env_name, str):
        try:
            env_name = RegisteredEnv(env_name)
        except ValueError:
            raise ValueError(f"Unknown environment: {env_name}")

    def _get_base_env(env_name: RegisteredEnv):
        if env_name == RegisteredEnv.TIC_TAC_TOE:
            return TicTacToe()
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
    env = _get_base_env(env_name)
    if auto_reset:
        env = AutoResetWrapper(env)

    return env
