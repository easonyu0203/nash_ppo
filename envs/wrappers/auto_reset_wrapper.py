from typing import Tuple
from functools import cached_property
import chex
import jax

from envs.mytypes import BaseEnv, EnvState, TimeStep, Action
from envs.myspaces import Space
from envs.wrappers.wrapper import Wrapper

class AutoResetWrapper(Wrapper):
    """
    Auto reset the env, with MODE=SAME_STEP
    """

    def __init__(self, env: BaseEnv):
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

    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        return self._env.reset(key)

    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        state, timestep = self._env.step(state, action)

        state, timestep = jax.lax.cond(
            timestep.done,
            self._auto_reset,
            lambda s, t: (s, t), # not done -> continue as normal
            state, timestep
        )

        return state, timestep
    
    def _auto_reset(self, state: EnvState, timestep: TimeStep) -> Tuple[EnvState, TimeStep]:
        """auto reset with mode=same_step"""
        new_state, new_timestep = self._env.reset(state.key)

        return new_state, TimeStep(
            reward=timestep.reward,
            done=timestep.done,
            observation=new_timestep.observation,
            action_mask=new_timestep.action_mask,
            step_cnt=new_timestep.step_cnt,
            info=timestep.info, # NOTE: we use terminated step info, this mean the new episode first step info is gone
        )