"""Mixin for automatic agent configuration tracking and checkpointing."""

from typing import Dict, Any
import inspect


class CheckpointableMixin:
    """Mixin that provides configuration tracking and checkpointing capabilities.

    This mixin automatically tracks constructor arguments to enable automatic
    serialization/deserialization of agent configurations for checkpointing.

    Usage:
        class MyAgent(nnx.Module, CheckpointableMixin):
            def __init__(self, key, arg1, arg2, kwarg1=default):
                # Call parent __init__ explicitly
                super().__init__()
                # CheckpointableMixin will automatically store these args
                ...

    Note: This is a mixin class and should be used with multiple inheritance.
          It should typically appear after the primary base class in the MRO.
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
            # Only store config at the most derived class (self.__class__)
            # to avoid issues with intermediate base classes
            if not hasattr(self, '_agent_config'):
                object.__setattr__(self, '_agent_config', config)
                try:
                    object.__setattr__(self, '_agent_class_name', self.__class__.get_agent_class_name())
                except ValueError:
                    # If class is not registered (e.g., intermediate base class), use class name
                    object.__setattr__(self, '_agent_class_name', self.__class__.__name__)

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
        from agents import get_agent_name_from_class
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
