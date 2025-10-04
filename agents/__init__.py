from agents.base_agent import BaseAgent
from agents.mlp_agent import MLPAgent

import chex
from omegaconf import DictConfig


# Registry mapping agent_name to agent class
_AGENT_REGISTRY = {
    "mlp": MLPAgent,
    # Add more agents here as needed
}


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

    if agent_name not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent name: '{agent_name}'. "
            f"Available agents: {list(_AGENT_REGISTRY.keys())}"
        )

    agent_cls = _AGENT_REGISTRY[agent_name]

    # config dict passing as key word arguments
    config_dict = dict(agent_config)
    config_dict.pop('agent_name', None)

    return agent_cls(key, **config_dict)


__all__ = [
    'BaseAgent',
    'create_agent',
]

if __name__ == "__main__":
    print(f"Available agents: {list(_AGENT_REGISTRY.keys())}")