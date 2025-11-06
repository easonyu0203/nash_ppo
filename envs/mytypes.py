import abc
from dataclasses import dataclass
import numpy as np
from typing import Any, Union, Dict, Optional
from functools import cached_property
from gymnasium import Space

# Type aliases for observations and actions
Observation = Union[np.ndarray, Dict[str, np.ndarray]]
Action = Union[np.ndarray, Dict[str, np.ndarray]]


@dataclass
class TimeStep:
    """Container for environment timestep information.

    Attributes:
        reward: Rewards for each agent, shape (num_agents,)
        terminated: Natural episode end flags per agent, shape (num_agents,)
        truncated: Artificial time limit flags per agent, shape (num_agents,)
        observation: Observations for each agent, shape (num_agents, *obs_shape)
        info: Additional information dictionary
    """
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    observation: Observation
    info: Dict[str, np.ndarray]


class BaseEnv(abc.ABC):
    """Abstract base class for multi-agent environments.

    This class uses ABC (Abstract Base Class) rather than Protocol because:
    1. Wrappers inherit from it to extend functionality
    2. It provides concrete implementations (render, close, context manager)
    3. It establishes a clear inheritance hierarchy

    Assumptions:
        1. All agents share the same action space and observation space
        2. Agents can terminate/truncate independently (per-agent flags)

    Note: If you need structural subtyping without inheritance, consider
          creating a separate Protocol for type checking.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @cached_property
    @abc.abstractmethod
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        ...

    @cached_property
    @abc.abstractmethod
    def action_space(self) -> Space:
        """Action space shared by all agents."""
        ...

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> Space:
        """Observation space shared by all agents."""
        ...

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> TimeStep:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for environment initialization
            options: Additional reset options

        Returns:
            Initial timestep
        """
        ...

    @abc.abstractmethod
    def step(self, action: Action) -> TimeStep:
        """Execute one timestep in the environment.

        Args:
            action: Actions for all agents

        Returns:
            Timestep after executing actions
        """
        ...

    @property
    def unwrapped(self) -> 'BaseEnv':
        """Return the base unwrapped environment."""
        return self

    def render(self) -> Any:
        """Render the environment (optional, not all envs support this)."""
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Perform any necessary cleanup."""
        pass

    def __enter__(self) -> 'BaseEnv':
        """Support context manager protocol."""
        return self

    def __exit__(self, *args) -> None:
        """Support context manager protocol - cleanup on exit."""
        self.close()