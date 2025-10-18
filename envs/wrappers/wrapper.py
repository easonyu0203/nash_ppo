from typing import Any, Dict
from functools import cached_property

from gymnasium import Space

from envs.mytypes import Action, BaseEnv, TimeStep


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

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Action) -> TimeStep:
        return self._env.step(action)

    @cached_property
    def observation_space(self) -> Space:
        return self._env.observation_space

    @cached_property
    def action_space(self) -> Space:
        return self._env.action_space

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        return self._env.close()

    def __enter__(self) -> 'Wrapper':
        return self

    def __exit__(self) -> None:
        self.close()


