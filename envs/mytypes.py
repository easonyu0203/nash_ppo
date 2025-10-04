import abc
import chex
from typing import Any, Tuple, Union, Dict
from functools import cached_property

from envs.myspaces import Space

Observation = Union[chex.Array, Dict[str, chex.Array]]
Action = Union[chex.Array, Dict[str, chex.Array]]

@chex.dataclass
class TimeStep:
    reward: chex.Array # (num_agents, )
    done: chex.Numeric # ()
    observation: Observation # (num_agents, *obs_shape, )
    action_mask: Action # (num_agents, *obs_shape, )
    step_cnt: chex.Numeric # ()
    info: Dict[str, chex.Array]


@chex.dataclass
class EnvState:
    """This act as 'base' class of EnvState, but we don't actually inherit from this"""
    key: chex.PRNGKey
    current_player: chex.Numeric
    done: chex.Numeric # when done == False, step should raise error
    step_cnt: chex.Numeric

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
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        pass

    @abc.abstractmethod
    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        pass

    @property
    def unwrapped(self) -> 'BaseEnv':
        return self

    def render(self, state: EnvState) -> Any:
        """Render frames of the environment for a given state.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        raise NotImplementedError("Render method not implemented for this environment.")

    def close(self) -> None:
        """Perform any necessary cleanup."""

    def __enter__(self) -> 'BaseEnv':
        return self

    def __exit__(self, *args: Any) -> None:
        """Calls :meth:`close()`."""
        self.close()