from typing import Any, Dict
from functools import cached_property
from dataclasses import replace

from gymnasium import Space

from envs.mytypes import BaseEnv, TimeStep, Action
from envs.wrappers.wrapper import Wrapper

class AutoResetWrapper(Wrapper):
    """
    Auto reset the env, with MODE=DEFER
    """

    def __init__(self, env: BaseEnv):
        self._need_reset = False
        self._env = env

    @cached_property
    def num_agents(self) -> int:
        return self._env.num_agents

    @cached_property
    def action_space(self) -> Space:
        return self._env.action_space

    @cached_property
    def observation_space(self) -> Space:
        return self._env.observation_space

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Action) -> TimeStep:
        if self._need_reset:
            self._need_reset = False
            return self.reset()

        timestep = self._env.step(action)

        # Reset only if all agents are done (either terminated or truncated)
        self._need_reset = timestep.terminated | timestep.truncated


        return timestep