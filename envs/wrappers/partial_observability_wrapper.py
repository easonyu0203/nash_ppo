"""
Wrapper for creating Partially Observable Markov Decision Processes (POMDPs).

This wrapper filters observations to include only specified dimensions,
enabling POMDP scenarios from fully observable environments.
"""

from typing import Any, Dict, List
from functools import cached_property
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box

from envs.mytypes import BaseEnv, TimeStep, Action
from envs.wrappers.wrapper import Wrapper


class PartialObservabilityWrapper(Wrapper):
    """
    Wrapper that filters observations to create partial observability.

    This wrapper modifies the observation space to include only specified
    dimensions, effectively creating a POMDP from a fully observable environment.

    Example:
        # Pendulum normally has 3 observations: [cos(theta), sin(theta), angular_velocity]
        # To hide angular velocity (dimension 2), use observable_dims=[0, 1]
        wrapper = PartialObservabilityWrapper(env, observable_dims=[0, 1])
    """

    def __init__(self, env: BaseEnv, observable_dims: List[int]):
        """
        Initialize the partial observability wrapper.

        Args:
            env: The base environment to wrap
            observable_dims: List of observation dimension indices to keep.
                           Must be valid indices for the environment's observation space.

        Raises:
            ValueError: If environment has unsupported observation space
            ValueError: If observable_dims contains invalid indices
            ValueError: If observable_dims is empty
        """
        super().__init__(env)

        # Validate observation space is Box
        obs_space = env.observation_space
        if not isinstance(obs_space, Box):
            raise ValueError(
                f"PartialObservabilityWrapper only supports Box observation spaces. "
                f"Got: {type(obs_space).__name__}"
            )

        # Validate observable_dims
        if not observable_dims:
            raise ValueError("observable_dims cannot be empty")

        # Get original observation shape (handle both flat and multi-dimensional)
        if len(obs_space.shape) == 1:
            max_dim = obs_space.shape[0]
        else:
            raise ValueError(
                f"PartialObservabilityWrapper only supports 1D Box spaces. "
                f"Got shape: {obs_space.shape}"
            )

        # Validate indices
        self._observable_dims = list(observable_dims)
        for dim in self._observable_dims:
            if not isinstance(dim, int):
                raise ValueError(f"All observable_dims must be integers. Got: {dim}")
            if dim < 0 or dim >= max_dim:
                raise ValueError(
                    f"Observable dimension {dim} is out of bounds. "
                    f"Valid range: [0, {max_dim})"
                )

        # Check for duplicates
        if len(set(self._observable_dims)) != len(self._observable_dims):
            raise ValueError(f"observable_dims contains duplicates: {self._observable_dims}")

        # Store original observation space for reference
        self._original_obs_space = obs_space

    def __repr__(self) -> str:
        return f"PartialObservabilityWrapper({self._env!r}, observable_dims={self._observable_dims})"

    @cached_property
    def observation_space(self) -> Space:
        """Return the filtered observation space with only observable dimensions."""
        orig_space = self._original_obs_space

        # Create new Box space with filtered dimensions
        new_shape = (len(self._observable_dims),)
        new_low = orig_space.low[self._observable_dims]
        new_high = orig_space.high[self._observable_dims]

        return Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=orig_space.dtype
        )

    def _filter_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Filter observation array to include only observable dimensions.

        Args:
            obs: Observation array with shape (num_agents, obs_dim) or (num_agents, ...)

        Returns:
            Filtered observation array with shape (num_agents, len(observable_dims))
        """
        # obs has shape (num_agents, obs_dim, ...)
        # We need to filter along the observation dimension (axis 1)
        return obs[:, self._observable_dims]

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        """Reset environment and filter observations."""
        timestep = self._env.reset(seed=seed, options=options)

        # Filter observations
        filtered_obs = self._filter_observation(timestep.observation)

        return TimeStep(
            reward=timestep.reward,
            terminated=timestep.terminated,
            truncated=timestep.truncated,
            observation=filtered_obs,
            info=timestep.info
        )

    def step(self, action: Action) -> TimeStep:
        """Step environment and filter observations."""
        timestep = self._env.step(action)

        # Filter observations
        filtered_obs = self._filter_observation(timestep.observation)

        return TimeStep(
            reward=timestep.reward,
            terminated=timestep.terminated,
            truncated=timestep.truncated,
            observation=filtered_obs,
            info=timestep.info
        )


__all__ = ["PartialObservabilityWrapper"]
