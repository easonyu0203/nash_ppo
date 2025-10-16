from typing import Any, Dict
from functools import cached_property

from gymnasium import Space

from envs.mytypes import BaseEnv, TimeStep, Action
from envs.wrappers.wrapper import Wrapper

class AutoResetWrapper(Wrapper):
    """
    Auto reset the env, with MODE=DEFERRED
    """

    def __init__(self, env: BaseEnv):
        self._env = env
        self._needs_reset = True

    @cached_property
    def num_agents(self) -> int:
        return self._env.num_agents

    @cached_property
    def action_space(self) -> Space:
        return self._env.action_space

    @cached_property
    def observation_space(self) -> Space:
        return self._env.observation_space

    def reset(self, seed: int = None, options: Dict[Any] = None) -> TimeStep:
        self._needs_reset = False
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Action) -> TimeStep:
        # Deferred reset: if previous episode ended, reset before stepping
        if self._needs_reset:
            self._env.reset()
            self._needs_reset = False

        timestep = self._env.step(action)

        # Mark that we need to reset on next step if episode is done
        if timestep.done.item():
            self._needs_reset = True

        return timestep