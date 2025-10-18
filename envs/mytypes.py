import abc
from dataclasses import dataclass
import numpy as np
from typing import Any, Union, Dict
from functools import cached_property
from gymnasium import Space

Observation = Union[np.ndarray, Dict[str, np.ndarray]]
Action = Union[np.ndarray, Dict[str, np.ndarray]]

@dataclass
class TimeStep:
    reward: np.ndarray # (num_agents, )
    done: np.ndarray # ()
    observation: Observation # (num_agents, *obs_shape, )
    action_mask: Action # (num_agents, *obs_shape, )
    info: Dict[str, np.ndarray]


class BaseEnv(abc.ABC):
    """
    Assumption:
        1) agents share same action space and observation space
        2) all agent terminate at same step
    """

    def __repr__(self) -> str:
        return "Environment"

    @cached_property
    @abc.abstractmethod
    def num_agents(self) -> int:
        pass

    @cached_property
    @abc.abstractmethod
    def action_space(self) -> Space:
        pass

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> Space:
        pass

    @abc.abstractmethod
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        pass

    @abc.abstractmethod
    def step(self, action: Action) -> TimeStep:
        pass

    @property
    def unwrapped(self) -> 'BaseEnv':
        return self

    def render(self) -> Any:
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Perform any necessary cleanup."""

    def __enter__(self) -> 'BaseEnv':
        return self

    def __exit__(self) -> None:
        """Calls :meth:`close()`."""
        self.close()