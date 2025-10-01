from agents.base_agent import BaseAgent
from agents.mlp_agent import MLPAgent
from enum import Enum
from typing import Union
import chex


class RegisteredAgent(Enum):
    TIC_TAC_TOE = "tic_tac_toe"


def create_agent(agent_name: Union[RegisteredAgent, str], key: chex.PRNGKey) -> BaseAgent:
    """Create an agent instance based on the registered agent name.
    
    Args:
        agent_name: The registered agent type to create (enum or string)
        rngs: JAX random number generator state for initialization
        
    Returns:
        An instance of the specified agent type
        
    Raises:
        ValueError: If the agent name is not recognized
    """
    # Convert string to enum if needed
    if isinstance(agent_name, str):
        try:
            agent_name = RegisteredAgent(agent_name)
        except ValueError:
            raise ValueError(f"Unknown agent: {agent_name}")
        
    if agent_name == RegisteredAgent.TIC_TAC_TOE:
        return MLPAgent(key, input_dim=9, output_dim=9)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    

__all__ = ['BaseAgent', 'RegisteredAgent', 'create_agent']