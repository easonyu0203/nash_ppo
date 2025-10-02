"""Mixin for automatic agent configuration tracking."""

from typing import Dict, Any
import inspect


class ConfigurableAgent:
    """Mixin that automatically tracks constructor arguments for checkpointing.

    This mixin intercepts the __init__ call to store all constructor arguments,
    enabling automatic serialization/deserialization of agent configurations.

    Usage:
        class MyAgent(BaseAgent, ConfigurableAgent):
            def __init__(self, key, arg1, arg2, kwarg1=default):
                # ConfigurableAgent will automatically store these args
                super().__init__(key)
                ...
    """

    _agent_config: Dict[str, Any]
    _agent_class_name: str

    def __init_subclass__(cls, **kwargs):
        """Hook that wraps subclass __init__ to capture arguments."""
        super().__init_subclass__(**kwargs)

        # Store original __init__
        original_init = cls.__init__

        def wrapped_init(self, key, *args, **kwargs):
            """Wrapped __init__ that captures and stores arguments."""
            # Get the signature of the original __init__
            sig = inspect.signature(original_init)

            # Bind arguments to parameters
            bound_args = sig.bind(self, key, *args, **kwargs)
            bound_args.apply_defaults()

            # Extract config (exclude 'self' and 'key')
            config = dict(bound_args.arguments)
            config.pop('self', None)
            config.pop('key', None)

            # Store config before calling original __init__
            # This ensures config is available even if __init__ fails
            object.__setattr__(self, '_agent_config', config)
            object.__setattr__(self, '_agent_class_name', cls.get_agent_class_name())

            # Call original __init__
            original_init(self, key, *args, **kwargs)

        # Replace __init__ with wrapped version
        cls.__init__ = wrapped_init

    @classmethod
    def get_agent_class_name(cls) -> str:
        """Get the registered name for this agent class.

        Returns:
            The registered agent name (e.g., "mlp")

        Raises:
            ValueError: If the agent class is not registered
        """
        from agents.registry import get_agent_name_from_class
        return get_agent_name_from_class(cls)

    def get_config(self) -> Dict[str, Any]:
        """Get the stored configuration dictionary.

        Returns:
            Dictionary of constructor arguments (excluding 'key')
        """
        return dict(self._agent_config)

    def get_class_name(self) -> str:
        """Get the registered class name.

        Returns:
            The registered agent name (e.g., "mlp")
        """
        return self._agent_class_name
