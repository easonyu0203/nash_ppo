from typing import Any, Dict, Optional
from functools import cached_property
import logging
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Discrete, Box

from envs.mytypes import BaseEnv, TimeStep, Action

# Module-level logger
logger = logging.getLogger(__name__)

# Track if we've already logged padding info (to avoid spam in vectorized envs)
_padding_info_logged = False


class PettingZooWrapper(BaseEnv):
    """
    Wrapper to adapt PettingZoo parallel environments to BaseEnv interface.

    Handles heterogeneous observation spaces by padding smaller observations to match
    the largest observation space. All agents must still share the same action space.

    Assumptions:
        - All agents share the same action space (enforced)
        - Environment uses PettingZoo's parallel API
        - Only Discrete action spaces are supported
        - Only Box observation spaces are supported (1D vectors or 3D images)
        - All agents terminate together (episode ends when any agent is done)
        - If agents have different observation shapes, they will be padded with zeros
    """

    def __init__(self, env):
        """
        Args:
            env: A PettingZoo parallel environment instance

        Raises:
            ValueError: If environment has unsupported observation or action space
        """
        self._env = env

        # Get initial state to inspect spaces
        self._env.reset()

        # Get agent list (in consistent order)
        self._agents = sorted(self._env.agents)
        self._num_agents = len(self._agents)

        # Get all agent spaces
        action_spaces = [self._env.action_space(agent) for agent in self._agents]
        obs_spaces = [self._env.observation_space(agent) for agent in self._agents]

        # Check action spaces are identical (required)
        ref_action_space = action_spaces[0]
        for i, action_space in enumerate(action_spaces):
            if not self._spaces_equal(ref_action_space, action_space):
                raise ValueError(
                    f"All agents must have the same action space. "
                    f"Agent {self._agents[0]} has {ref_action_space}, "
                    f"but agent {self._agents[i]} has {action_space}"
                )

        # Validate supported space types
        if not isinstance(ref_action_space, Discrete):
            raise ValueError(
                f"Only Discrete action spaces are supported. "
                f"Got: {type(ref_action_space).__name__}"
            )

        # Check all observation spaces are Box type
        for i, obs_space in enumerate(obs_spaces):
            if not isinstance(obs_space, Box):
                raise ValueError(
                    f"Only Box observation spaces are supported. "
                    f"Agent {self._agents[i]} has {type(obs_space).__name__}"
                )

        # Handle potentially heterogeneous observation spaces
        # Store individual obs spaces for each agent
        self._agent_obs_spaces = {agent: obs_space for agent, obs_space in zip(self._agents, obs_spaces)}

        # Determine if padding is needed and compute unified observation space
        self._needs_padding, self._observation_space = self._compute_unified_obs_space(obs_spaces)

        # Log padding info only once (useful for vectorized envs)
        if self._needs_padding:
            global _padding_info_logged
            if not _padding_info_logged:
                logger.info(
                    "Detected heterogeneous observation spaces. Will pad to shape %s",
                    self._observation_space.shape
                )
                for agent, obs_space in self._agent_obs_spaces.items():
                    logger.info("  %s: %s -> %s", agent, obs_space.shape, self._observation_space.shape)
                _padding_info_logged = True

        self._action_space = ref_action_space

        super().__init__()

    def _compute_unified_obs_space(self, obs_spaces):
        """
        Compute a unified observation space that can accommodate all agent observations.

        Returns:
            (needs_padding, unified_space): Tuple of whether padding is needed and the unified space
        """
        # Check if all spaces are identical
        ref_space = obs_spaces[0]
        all_same = all(self._spaces_equal(ref_space, space) for space in obs_spaces)

        if all_same:
            return False, ref_space

        # Spaces differ - need to pad
        # We only support 1D (vector) or 3D (image) observations
        shapes = [space.shape for space in obs_spaces]
        ndims = [len(shape) for shape in shapes]

        if not all(ndim == ndims[0] for ndim in ndims):
            raise ValueError(
                f"All observation spaces must have the same number of dimensions. "
                f"Got shapes: {shapes}"
            )

        ndim = ndims[0]

        if ndim == 1:
            # 1D vector observations - pad to max length
            max_dim = max(shape[0] for shape in shapes)
            unified_shape = (max_dim,)
        elif ndim == 3:
            # 3D image observations (H, W, C) - pad each dimension to max
            max_h = max(shape[0] for shape in shapes)
            max_w = max(shape[1] for shape in shapes)
            max_c = max(shape[2] for shape in shapes)
            unified_shape = (max_h, max_w, max_c)
        else:
            raise ValueError(
                f"Only 1D (vector) or 3D (image) observation spaces are supported. "
                f"Got shape with {ndim} dimensions: {shapes[0]}"
            )

        # Create unified Box space
        # Use the most permissive bounds
        low = min(space.low.min() for space in obs_spaces)
        high = max(space.high.max() for space in obs_spaces)
        dtype = obs_spaces[0].dtype  # Assume all have same dtype

        unified_space = Box(low=low, high=high, shape=unified_shape, dtype=dtype)
        return True, unified_space

    def _spaces_equal(self, space1: Space, space2: Space) -> bool:
        """Check if two spaces are equal."""
        if type(space1) != type(space2):
            return False

        if isinstance(space1, Discrete):
            return space1.n == space2.n
        elif isinstance(space1, Box):
            return (
                np.array_equal(space1.shape, space2.shape) and
                np.array_equal(space1.low, space2.low) and
                np.array_equal(space1.high, space2.high) and
                space1.dtype == space2.dtype
            )

        return False

    def __repr__(self) -> str:
        return f"PettingZooWrapper({self._env!r})"

    @cached_property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return self._num_agents

    @cached_property
    def observation_space(self) -> Space:
        return self._observation_space

    @cached_property
    def action_space(self) -> Space:
        return self._action_space

    @cached_property
    def _default_action_mask(self) -> Action:
        """
        Generate default action mask (all actions available).
        Returns array with shape (num_agents, num_actions).
        """
        return np.ones((self.num_agents, self.action_space.n), dtype=np.int8)

    def _dict_to_array(self, agent_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert dict of agent observations/rewards to array.
        Maintains consistent ordering using self._agents.

        Args:
            agent_dict: Dict mapping agent_id -> value

        Returns:
            Array with values ordered by self._agents
        """
        return np.array([agent_dict[agent] for agent in self._agents])

    def _pad_observation(self, obs: np.ndarray, agent: str) -> np.ndarray:
        """
        Pad an observation to match the unified observation space.

        Args:
            obs: Observation from the environment
            agent: Agent identifier

        Returns:
            Padded observation
        """
        if not self._needs_padding:
            return obs

        target_shape = self._observation_space.shape
        current_shape = obs.shape

        # Create padded array filled with zeros
        padded = np.zeros(target_shape, dtype=obs.dtype)

        if len(target_shape) == 1:
            # 1D vector - copy data to beginning
            padded[:current_shape[0]] = obs
        elif len(target_shape) == 3:
            # 3D image - copy data to top-left corner
            h, w, c = current_shape
            padded[:h, :w, :c] = obs
        else:
            raise ValueError(f"Unsupported observation shape: {current_shape}")

        return padded

    def _pad_observations(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert dict of observations to padded array.

        Args:
            obs_dict: Dict mapping agent_id -> observation

        Returns:
            Padded observation array of shape (num_agents, *unified_obs_shape)
        """
        padded_obs = []
        for agent in self._agents:
            obs = obs_dict[agent]
            padded = self._pad_observation(obs, agent)
            padded_obs.append(padded)
        return np.array(padded_obs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> TimeStep:
        """Reset the environment and return initial timestep."""
        # PettingZoo parallel API
        observations, infos = self._env.reset(seed=seed, options=options)

        # Convert observations dict to array with padding if needed (num_agents, *obs_shape)
        obs_array = self._pad_observations(observations)

        return TimeStep(
            reward=np.zeros(self.num_agents, dtype=np.float32),
            terminated=np.zeros(self.num_agents, dtype=bool),
            truncated=np.zeros(self.num_agents, dtype=bool),
            observation=obs_array,
            action_mask=self._default_action_mask,
            info=infos,
        )

    def step(self, action: Action) -> TimeStep:
        """
        Step the environment with the given action.

        Args:
            action: Action array with shape (num_agents,)
        """
        # Convert action array to dict
        action_dict = {agent: int(action[i]) for i, agent in enumerate(self._agents)}

        # Step environment
        observations, rewards, terminations, truncations, infos = self._env.step(action_dict)

        # Convert to arrays with padding if needed
        obs_array = self._pad_observations(observations)
        reward_array = self._dict_to_array(rewards)

        # Create per-agent terminated and truncated arrays
        terminated_array = np.array([
            terminations.get(agent, False)
            for agent in self._agents
        ], dtype=bool)

        truncated_array = np.array([
            truncations.get(agent, False)
            for agent in self._agents
        ], dtype=bool)

        return TimeStep(
            reward=reward_array.astype(np.float32),
            terminated=terminated_array,
            truncated=truncated_array,
            observation=obs_array,
            action_mask=self._default_action_mask,
            info=infos,
        )

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def unwrapped(self) -> BaseEnv:
        """Returns the unwrapped environment."""
        return self
