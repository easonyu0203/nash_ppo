from typing import Any, Dict, Optional
from functools import cached_property
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box

from envs.mytypes import BaseEnv, TimeStep, Action
from envs.wrappers.wrapper import Wrapper


class AddAgentIDWrapper(Wrapper):
    """
    Wrapper that adds agent ID information to observations.

    For vector observations (1D Box), appends one-hot encoded agent ID.
    For image observations (3D Box with shape H,W,C), adds agent ID as additional channels.

    This allows agents to distinguish themselves in symmetric multi-agent environments.
    """

    def __init__(self, env: BaseEnv, mode: str = "auto"):
        """
        Args:
            env: Base environment to wrap
            mode: How to add agent ID:
                - "auto": Automatically detect based on observation space shape
                - "vector": Append one-hot ID to vector observations (1D or flattened)
                - "image": Add ID as additional image channels (3D observations)

        Raises:
            ValueError: If mode is invalid or incompatible with observation space
        """
        super().__init__(env)

        if mode not in ["auto", "vector", "image"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto', 'vector', or 'image'")

        # Determine mode based on observation space if auto
        obs_space = env.observation_space
        if not isinstance(obs_space, Box):
            raise ValueError(
                f"AddAgentIDWrapper only supports Box observation spaces. "
                f"Got: {type(obs_space).__name__}"
            )

        obs_shape = obs_space.shape

        if mode == "auto":
            # Heuristic: 3D shape with last dim small (< 10) is likely image (H, W, C)
            # Otherwise treat as vector
            if len(obs_shape) == 3 and obs_shape[-1] < 10:
                mode = "image"
            else:
                mode = "vector"

        self._mode = mode

        # Create modified observation space
        if mode == "vector":
            # Flatten original space and append num_agents dimensions for one-hot ID
            original_dim = int(np.prod(obs_shape))
            new_dim = original_dim + env.num_agents
            self._observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(new_dim,),
                dtype=np.float32
            )
        else:  # image mode
            # Add num_agents channels to image
            if len(obs_shape) != 3:
                raise ValueError(
                    f"Image mode requires 3D observations (H, W, C). "
                    f"Got shape: {obs_shape}"
                )
            h, w, c = obs_shape
            new_c = c + env.num_agents
            self._observation_space = Box(
                low=obs_space.low.min(),
                high=obs_space.high.max(),
                shape=(h, w, new_c),
                dtype=obs_space.dtype
            )

    def __repr__(self) -> str:
        return f"AddAgentIDWrapper({self._env!r}, mode={self._mode})"

    @cached_property
    def observation_space(self) -> Space:
        return self._observation_space

    def _add_agent_ids(self, observations: np.ndarray) -> np.ndarray:
        """
        Add agent IDs to observations.

        Handles both flat and batched observations:
        - Flat: (num_agents, *obs_shape)
        - Batched: (batch_dim, num_agents, *obs_shape) - e.g., Unity's (num_areas, num_agents_per_area, obs_dim)

        Args:
            observations: Array of shape (num_agents, *obs_shape) or (batch_dim, num_agents, *obs_shape)

        Returns:
            Modified observations with agent IDs added
        """
        # Check if observations are batched (3+ dimensions for vector, 4+ for image)
        is_batched = (self._mode == "vector" and observations.ndim >= 3) or \
                     (self._mode == "image" and observations.ndim >= 4)

        if is_batched:
            # Handle batched observations (e.g., Unity's multi-area structure)
            # Shape: (batch_dim, num_agents_per_batch, *obs_shape)
            batch_dim = observations.shape[0]
            num_agents = observations.shape[1]

            if self._mode == "vector":
                # Flatten observations per agent: (batch_dim, num_agents, obs_dim)
                flat_obs = observations.reshape(batch_dim, num_agents, -1)

                # Create one-hot agent IDs: (num_agents, num_agents)
                # Broadcast to: (batch_dim, num_agents, num_agents)
                agent_ids = np.eye(num_agents, dtype=np.float32)
                agent_ids = np.broadcast_to(agent_ids, (batch_dim, num_agents, num_agents))

                # Concatenate: (batch_dim, num_agents, obs_dim + num_agents)
                return np.concatenate([flat_obs, agent_ids], axis=-1)

            else:  # image mode
                # Original shape: (batch_dim, num_agents, H, W, C)
                h, w, c = observations.shape[2:]

                # Create agent ID channels: (batch_dim, num_agents, H, W, num_agents)
                id_channels = np.zeros((batch_dim, num_agents, h, w, num_agents), dtype=observations.dtype)
                for i in range(num_agents):
                    id_channels[:, i, :, :, i] = 1.0

                # Concatenate: (batch_dim, num_agents, H, W, C + num_agents)
                return np.concatenate([observations, id_channels], axis=-1)

        else:
            # Handle flat observations (original behavior)
            num_agents = observations.shape[0]

            if self._mode == "vector":
                # Flatten observations and append one-hot agent IDs
                # Original shape: (num_agents, *obs_shape)
                # Flatten to: (num_agents, obs_dim)
                flat_obs = observations.reshape(num_agents, -1)

                # Create one-hot agent IDs: (num_agents, num_agents)
                agent_ids = np.eye(num_agents, dtype=np.float32)

                # Concatenate: (num_agents, obs_dim + num_agents)
                return np.concatenate([flat_obs, agent_ids], axis=-1)

            else:  # image mode
                # Add agent ID as additional channels
                # Original shape: (num_agents, H, W, C)
                h, w, c = observations.shape[1:]

                # Create agent ID channels: (num_agents, H, W, num_agents)
                # Each agent gets a channel filled with 1.0 for their ID, 0.0 for others
                id_channels = np.zeros((num_agents, h, w, num_agents), dtype=observations.dtype)
                for i in range(num_agents):
                    id_channels[i, :, :, i] = 1.0

                # Concatenate along channel dimension: (num_agents, H, W, C + num_agents)
                return np.concatenate([observations, id_channels], axis=-1)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> TimeStep:
        """Reset environment and add agent IDs to observations."""
        timestep = self._env.reset(seed=seed, options=options)

        # Add agent IDs to observations
        modified_obs = self._add_agent_ids(timestep.observation)

        return TimeStep(
            reward=timestep.reward,
            terminated=timestep.terminated,
            truncated=timestep.truncated,
            observation=modified_obs,
            info=timestep.info,
        )

    def step(self, action: Action) -> TimeStep:
        """Step environment and add agent IDs to observations."""
        timestep = self._env.step(action)

        # Add agent IDs to observations
        modified_obs = self._add_agent_ids(timestep.observation)

        return TimeStep(
            reward=timestep.reward,
            terminated=timestep.terminated,
            truncated=timestep.truncated,
            observation=modified_obs,
            info=timestep.info,
        )
