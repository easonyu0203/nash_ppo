"""Agent registration system for checkpoint management."""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent

# Bidirectional mappings for agent class <-> name
_AGENT_CLASS_TO_NAME: Dict[Type["BaseAgent"], str] = {}
_AGENT_NAME_TO_CLASS: Dict[str, Type["BaseAgent"]] = {}


def register_agent(name: str):
    """Decorator to register an agent class with a unique name.

    This enables bidirectional lookup between agent classes and their string identifiers,
    which is essential for saving/loading checkpoints.

    Args:
        name: Unique identifier for the agent class (e.g., "mlp")

    Returns:
        Decorator function that registers the class

    Example:
        @register_agent("mlp")
        class MLPAgent(BaseAgent, ConfigurableAgent):
            ...
    """
    def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        if name in _AGENT_NAME_TO_CLASS:
            existing_cls = _AGENT_NAME_TO_CLASS[name]
            if existing_cls is not cls:
                raise ValueError(
                    f"Agent name '{name}' is already registered to {existing_cls.__name__}"
                )

        _AGENT_CLASS_TO_NAME[cls] = name
        _AGENT_NAME_TO_CLASS[name] = cls
        return cls

    return decorator


def get_agent_class_from_name(name: str) -> Type["BaseAgent"]:
    """Get agent class from its registered name.

    Args:
        name: Registered agent name (e.g., "mlp")

    Returns:
        The agent class

    Raises:
        ValueError: If the agent name is not registered
    """
    if name not in _AGENT_NAME_TO_CLASS:
        raise ValueError(
            f"Unknown agent name: '{name}'. "
            f"Available agents: {list(_AGENT_NAME_TO_CLASS.keys())}"
        )
    return _AGENT_NAME_TO_CLASS[name]


def get_agent_name_from_class(cls: Type["BaseAgent"]) -> str:
    """Get registered name from agent class.

    Args:
        cls: Agent class

    Returns:
        The registered agent name

    Raises:
        ValueError: If the agent class is not registered
    """
    if cls not in _AGENT_CLASS_TO_NAME:
        raise ValueError(
            f"Agent class {cls.__name__} is not registered. "
            f"Use @register_agent decorator to register it."
        )
    return _AGENT_CLASS_TO_NAME[cls]


def list_registered_agents() -> Dict[str, Type["BaseAgent"]]:
    """Get all registered agents.

    Returns:
        Dictionary mapping agent names to their classes
    """
    return dict(_AGENT_NAME_TO_CLASS)
