from agents.base_agent import BaseAgent
from agents.registry import get_agent_class_from_name, list_registered_agents

# NOTE: Import all agent implementations to trigger registration
from agents.mlp_agent import MLPAgent

import chex
from omegaconf import DictConfig


def create_agent(agent_config: DictConfig, key: chex.PRNGKey) -> BaseAgent:
    """Create an agent instance based on the registered agent name.

    Args:
        agent_config: The config object with `agent_name` and key word parameters
        key: JAX random number generator state for initialization

    Returns:
        An instance of the specified agent type

    Raises:
        ValueError: If the agent name is not recognized
    """
    agent_cls = get_agent_class_from_name(agent_config.agent_name)

    # config dict passing as key word arguments
    config_dict = dict(agent_config)
    config_dict.pop('agent_name', None)

    return agent_cls(key, **config_dict)


__all__ = [
    'BaseAgent',
    'create_agent',
    'list_registered_agents'
]

if __name__ == "__main__":
    print(list_registered_agents())