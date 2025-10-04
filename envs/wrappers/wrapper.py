from typing import Any, Tuple
import chex
from functools import cached_property

from envs.myspaces import Space
from envs.mytypes import Action, BaseEnv, EnvState, TimeStep


class Wrapper(BaseEnv):
    """Base Wrapper class"""

    def __init__(self, env: BaseEnv):
        self._env = env
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._env!r})"

    @property
    def unwrapped(self) -> BaseEnv:
        """Returns the wrapped env."""
        return self._env.unwrapped

    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        return self._env.reset(key)

    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        return self._env.step(state, action)

    @cached_property
    def observation_space(self) -> Space:
        return self._env.observation_space

    @cached_property
    def action_space(self) -> Space:
        return self._env.action_space

    def render(self, state: EnvState) -> Any:
        return self._env.render(state)

    def close(self) -> None:
        """Perform any necessary cleanup.
        """
        return self._env.close()

    def __enter__(self) -> 'Wrapper':
        return self

    def __exit__(self, *args: Any) -> None:
        del args  # Unused
        self.close()


