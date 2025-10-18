from agents.base_agent import BaseAgent
from agents.mlp_agent import MLPAgent
from agents.image_agent import ImageAgent

import chex
from omegaconf import DictConfig, OmegaConf
from typing import Type


# Registry mapping agent_name to agent class
_AGENT_REGISTRY = {
    "mlp": MLPAgent,
    "image": ImageAgent,
    # Add more agents here as needed
}


def get_agent_class_from_name(name: str) -> Type[BaseAgent]:
    """Get agent class from registered name.

    Args:
        name: Registered agent name (e.g., "mlp")

    Returns:
        Agent class

    Raises:
        ValueError: If agent name is not registered
    """
    if name not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent name: '{name}'. "
            f"Available agents: {list(_AGENT_REGISTRY.keys())}"
        )
    return _AGENT_REGISTRY[name]


def get_agent_name_from_class(cls: Type[BaseAgent]) -> str:
    """Get registered name from agent class.

    Args:
        cls: Agent class

    Returns:
        Registered agent name (e.g., "mlp")

    Raises:
        ValueError: If agent class is not registered
    """
    for name, agent_cls in _AGENT_REGISTRY.items():
        if agent_cls is cls:
            return name
    raise ValueError(
        f"Agent class {cls.__name__} is not registered. "
        f"Available agents: {list(_AGENT_REGISTRY.keys())}"
    )


def create_agent(agent_config: DictConfig, key: chex.PRNGKey) -> BaseAgent:
    """Create an agent instance based on the config.

    Args:
        agent_config: The config object with `agent_name` and key word parameters
        key: JAX random number generator state for initialization

    Returns:
        An instance of the specified agent type

    Raises:
        ValueError: If the agent name is not recognized
    """
    agent_name = agent_config.agent_name
    agent_cls = get_agent_class_from_name(agent_name)

    # Convert OmegaConf to standard Python types (dict, list, str, int, float)
    # This ensures agents receive standard types, not ListConfig/DictConfig
    config_dict = OmegaConf.to_container(agent_config, resolve=True)
    config_dict.pop('agent_name', None)

    return agent_cls(key, **config_dict)


__all__ = [
    'BaseAgent',
    'create_agent',
    'get_agent_class_from_name',
    'get_agent_name_from_class',
]

if __name__ == "__main__":
    print(f"Available agents: {list(_AGENT_REGISTRY.keys())}")