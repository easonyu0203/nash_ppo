from functools import cached_property
from typing import Tuple
import chex
import jax.numpy as jnp
from jumanji import specs

from envs.wrappers.wrapper import Wrapper
from envs.myspaces import Space, Discrete, Box
from envs.mytypes import EnvState, TimeStep, Action


class JumanjiWrapper(Wrapper):

    @cached_property
    def num_agents(self) -> int:
        return self._env.num_agents

    @cached_property
    def action_space(self) -> Space:
        """Convert to our space specification"""

        action_spec = self._env.action_spec
        if not isinstance(action_spec, specs.MultiDiscreteArray):
            raise ValueError("only support Jumanji env with discrete action space")

        num_values = action_spec._num_values
        if len(num_values) != self.num_agents:
            raise ValueError("number of discrete Array should be the same as number of agents")

        # Only support all agents have same action space
        if any(v != num_values[0] for v in num_values):
            raise ValueError("Only support all agent with same action space")

        return Discrete(num_categories=num_values[0])

    @cached_property
    def observation_space(self) -> Space:
        """Convert to our space specification"""
        obs_spec = self._env.observation_spec
        agents_view = obs_spec.agents_view

        # Only support Array or BoundedArray
        if not isinstance(agents_view, (specs.Array, specs.BoundedArray)):
            raise ValueError(f"Only support Array or BoundedArray, got {type(agents_view)}")

        # Check first dimension matches num_agents
        if agents_view.shape[0] != self.num_agents:
            raise ValueError(
                f"First dimension should be num_agents ({self.num_agents}), got {agents_view.shape[0]}"
            )

        # Return space for single agent (remove first dimension)
        single_agent_shape = agents_view.shape[1:]

        if isinstance(agents_view, specs.BoundedArray):
            low = agents_view.minimum[0] if len(agents_view.minimum.shape) > 0 else agents_view.minimum
            high = agents_view.maximum[0] if len(agents_view.maximum.shape) > 0 else agents_view.maximum
            return Box(low=float(low), high=float(high), shape=single_agent_shape, dtype=agents_view.dtype)
        else:  # Array
            return Box(low=-1000.0, high=1000.0, shape=single_agent_shape, dtype=agents_view.dtype)



    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        state, jumanji_timestep = self._env.reset(key)

        # Broadcast scalar reward to per-agent shape (num_agents,)
        reward = jumanji_timestep.reward
        if reward.ndim == 0:  # scalar reward
            reward = jnp.full((self.num_agents,), reward)

        timestep = TimeStep(
            observation=jumanji_timestep.observation.agents_view,
            action_mask=jumanji_timestep.observation.action_mask,
            reward=reward,
            done=jumanji_timestep.last(),
            step_cnt=jumanji_timestep.observation.step_count,
            info=jumanji_timestep.extras
        )

        return state, timestep

    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        state, jumanji_timestep = self._env.step(state, action)

        # Broadcast scalar reward to per-agent shape (num_agents,)
        reward = jumanji_timestep.reward
        if reward.ndim == 0:  # scalar reward
            reward = jnp.full((self.num_agents,), reward)

        timestep = TimeStep(
            observation=jumanji_timestep.observation.agents_view,
            action_mask=jumanji_timestep.observation.action_mask,
            reward=reward,
            done=jumanji_timestep.last(),
            step_cnt=jumanji_timestep.observation.step_count,
            info=jumanji_timestep.extras
        )

        return state, timestep
