from abc import ABC, abstractmethod
from functools import cached_property
from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.routing.connector.constants import (
    AGENT_INITIAL_VALUE,
)

from envs.wrappers.wrapper import Wrapper
from envs.myspaces import Space, Discrete, Box
from envs.mytypes import EnvState, TimeStep, Action


class JumanjiWrapper(Wrapper, ABC):
    """Base wrapper for Jumanji environments.

    Our API expects:
    - observation: (num_agents, *obs_shape)
    - action_mask: (num_agents, *action_shape)
    - reward: (num_agents,)
    - done: scalar
    - step_cnt: scalar
    """

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

    @abstractmethod
    def _get_observation_space_spec(self) -> specs.Spec:
        """Get the observation space spec for agents_view.

        Different environments may structure observations differently.
        """
        pass

    @cached_property
    def observation_space(self) -> Space:
        """Convert to our space specification"""
        agents_view_spec = self._get_observation_space_spec()

        # Only support Array or BoundedArray
        if not isinstance(agents_view_spec, (specs.Array, specs.BoundedArray)):
            raise ValueError(f"Only support Array or BoundedArray, got {type(agents_view_spec)}")

        # Check first dimension matches num_agents
        if agents_view_spec.shape[0] != self.num_agents:
            raise ValueError(
                f"First dimension should be num_agents ({self.num_agents}), got {agents_view_spec.shape[0]}"
            )

        # Return space for single agent (remove first dimension)
        single_agent_shape = agents_view_spec.shape[1:]

        if isinstance(agents_view_spec, specs.BoundedArray):
            low = agents_view_spec.minimum[0] if len(agents_view_spec.minimum.shape) > 0 else agents_view_spec.minimum
            high = agents_view_spec.maximum[0] if len(agents_view_spec.maximum.shape) > 0 else agents_view_spec.maximum
            return Box(low=float(low), high=float(high), shape=single_agent_shape, dtype=agents_view_spec.dtype)
        elif isinstance(agents_view_spec, specs.Array):  # Array
            return Box(low=-1000.0, high=1000.0, shape=single_agent_shape, dtype=agents_view_spec.dtype)
        else:
            raise ValueError(f"don't suport this spec for jumanji env:\n {agents_view_spec}")

    @abstractmethod
    def _extract_agents_view(self, jumanji_timestep) -> chex.Array:
        """Extract per-agent observations (num_agents, *obs_shape)."""
        pass

    @abstractmethod
    def _extract_reward(self, jumanji_timestep) -> chex.Array:
        """Extract per-agent rewards (num_agents,)."""
        pass

    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        state, jumanji_timestep = self._env.reset(key)

        timestep = TimeStep(
            observation=self._extract_agents_view(jumanji_timestep),
            action_mask=jumanji_timestep.observation.action_mask,
            reward=self._extract_reward(jumanji_timestep),
            done=jumanji_timestep.last(),
            step_cnt=jumanji_timestep.observation.step_count,
            info=jumanji_timestep.extras
        )

        return state, timestep

    def step(self, state: EnvState, action: Action) -> Tuple[EnvState, TimeStep]:
        state, jumanji_timestep = self._env.step(state, action)

        timestep = TimeStep(
            observation=self._extract_agents_view(jumanji_timestep),
            action_mask=jumanji_timestep.observation.action_mask,
            reward=self._extract_reward(jumanji_timestep),
            done=jumanji_timestep.last(),
            step_cnt=jumanji_timestep.observation.step_count,
            info=jumanji_timestep.extras
        )

        return state, timestep


class RobotWarehouseWrapper(JumanjiWrapper):
    """Wrapper for Jumanji RobotWarehouse environment."""

    def _get_observation_space_spec(self) -> specs.Spec:
        return self._env.observation_spec.agents_view

    def _extract_agents_view(self, jumanji_timestep) -> chex.Array:
        # RobotWarehouse already has agents_view in the right format
        return jumanji_timestep.observation.agents_view

    def _extract_reward(self, jumanji_timestep) -> chex.Array:
        # RobotWarehouse has scalar reward, broadcast to (num_agents,)
        reward = jumanji_timestep.reward
        return jnp.full((self.num_agents,), reward)


class ConnectorWrapper(JumanjiWrapper):
    """Wrapper for Jumanji Connector environment."""

    def __init__(self, env):
        super().__init__(env)
        self.agent_ids = jnp.arange(self.num_agents)
        self.grid_size = self._env.grid_size

    def _get_observation_space_spec(self) -> specs.Spec:
        # Connector transforms grid to agents_view with shape (num_agents, grid_size, grid_size)
        # Single channel with integer values that encode both agent ID and cell type
        return specs.BoundedArray(
            shape=(self.num_agents, self.grid_size, self.grid_size),
            dtype=jnp.int32,
            name="agents_view",
            minimum=0,
            maximum=3 * self.num_agents,
        )

    def _extract_agents_view(self, jumanji_timestep) -> chex.Array:
        # Transform grid to agents_view (num_agents, grid_size, grid_size)
        # Single channel where values encode both agent ID and cell type:
        # - EMPTY (0): empty cell
        # - agent_id * TARGET + PATH: path cell for agent_id
        # - agent_id * TARGET + POSITION: position cell for agent_id
        # - agent_id * TARGET: target cell for agent_id
        grid = jumanji_timestep.observation.grid

        # Switch perspective for each agent
        agents_view = jax.vmap(self._switch_perspective, in_axes=(None, 0, None))(
            grid, self.agent_ids, self.num_agents
        )

        return agents_view

    def _extract_reward(self, jumanji_timestep) -> chex.Array:
        # Connector already has per-agent rewards (num_agents,)
        return jumanji_timestep.reward

    @staticmethod
    def _switch_perspective(grid: chex.Array, agent_id: int, num_agents: int) -> chex.Array:
        """Switch grid perspective to be relative to the given agent."""
        new_grid = grid - AGENT_INITIAL_VALUE  # Center agent values around 0
        new_grid -= 3 * agent_id  # Rotate perspective
        new_grid %= 3 * num_agents  # Keep in bounds
        new_grid += AGENT_INITIAL_VALUE  # Un-center
        # Take agent values from rotated grid and empty values from old grid
        return jnp.where((grid >= AGENT_INITIAL_VALUE), new_grid, grid)
