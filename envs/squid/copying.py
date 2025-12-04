"""Copying task environment for sequence memory research.

This environment is similar to copying tasks used in LLM research, where an agent
must remember and recall a sequence of tokens after seeing a prompt.
"""

import numpy as np
from functools import cached_property
from typing import Optional, Dict, Any
from gymnasium.spaces import Discrete, Box

from envs.mytypes import BaseEnv, TimeStep


class CopyingEnv(BaseEnv):
    """Copying task environment.

    The agent observes a sequence of tokens, then must recall them after seeing
    a prompt token. The task tests the agent's ability to remember sequences.

    Episode structure:
    1. First x steps (or scattered): Agent observes the "sentence" (tokens to remember)
    2. Middle steps: White noise tokens (distractors)
    3. Last x steps (recall phase): All observations are prompt token, agent must output sentence

    Rewards:
    - +1 for each correctly recalled token during recall phase
    - 0 for all other timesteps

    Args:
        sequence_length: Total length of the episode (must be >= 2 * num_tokens_to_remember)
        vocab_size: Size of token vocabulary (includes white noise and prompt tokens)
        num_tokens_to_remember: Length of sentence to remember (x)
        selective_copy: If True, sentence tokens appear with random intervals
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
        num_tokens_to_remember: int,
        selective_copy: bool = False,
        seed: Optional[int] = None,
    ):
        if sequence_length < 2 * num_tokens_to_remember:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be >= 2 * num_tokens_to_remember "
                f"({2 * num_tokens_to_remember})"
            )

        if vocab_size < 3:
            raise ValueError(
                f"vocab_size must be at least 3 (sentence tokens + white noise + prompt), "
                f"got {vocab_size}"
            )

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_tokens_to_remember = num_tokens_to_remember
        self.selective_copy = selective_copy

        # Reserve special tokens
        self.white_noise_token = vocab_size - 2
        self.prompt_token = vocab_size - 1
        self.max_sentence_token = vocab_size - 3  # 0 to vocab_size-3 are valid sentence tokens

        # Episode state
        self._rng = np.random.RandomState(seed)
        self._sentence: Optional[np.ndarray] = None
        self._step_count = 0
        self._obs_sequence: Optional[np.ndarray] = None
        self._recall_start = sequence_length - num_tokens_to_remember

    @cached_property
    def num_agents(self) -> int:
        """Single agent environment."""
        return 1

    @cached_property
    def action_space(self) -> Discrete:
        """Action space is discrete tokens."""
        return Discrete(self.vocab_size)

    @cached_property
    def observation_space(self) -> Box:
        """Observation space is discrete tokens with a dummy dimension."""
        return Box(low=0, high=self.vocab_size - 1, shape=(1,), dtype=np.int32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> TimeStep:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for environment initialization
            options: Additional reset options (unused)

        Returns:
            Initial timestep with first observation
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Generate random sentence to remember
        self._sentence = self._rng.randint(0, self.max_sentence_token + 1, size=self.num_tokens_to_remember)
        self._step_count = 0

        # Precompute entire observation sequence
        self._obs_sequence = np.full(self.sequence_length, self.white_noise_token, dtype=np.int32)

        if self.selective_copy:
            # Tokens appear at random intervals before recall phase
            positions = self._rng.choice(self._recall_start, size=self.num_tokens_to_remember, replace=False)
            self._obs_sequence[positions] = self._sentence
        else:
            # Tokens appear sequentially at the beginning
            self._obs_sequence[:self.num_tokens_to_remember] = self._sentence

        # Set all recall phase observations to prompt token
        self._obs_sequence[self._recall_start:] = self.prompt_token

        return TimeStep(
            reward=np.array([0.0]),
            terminated=np.array([False]),
            truncated=np.array([False]),
            observation=np.array([[self._obs_sequence[0]]], dtype=np.int32),
            info={"sentence": self._sentence.copy()}
        )

    def step(self, action: np.ndarray) -> TimeStep:
        """Execute one timestep in the environment.

        Args:
            action: Action for the agent (token to output), shape (1,) or scalar

        Returns:
            Timestep after executing action
        """
        # Handle both scalar and array actions
        if isinstance(action, np.ndarray):
            action_value = int(action.item()) if action.size == 1 else int(action[0])
        else:
            action_value = int(action)

        # Calculate reward if in recall phase
        reward = 0.0
        if self._step_count >= self._recall_start:
            recall_index = self._step_count - self._recall_start
            if action_value == self._sentence[recall_index]:
                reward = 1.0

        self._step_count += 1

        # Check if episode is done
        terminated = self._step_count >= self.sequence_length

        # Get observation (or white noise if terminated)
        obs = self.white_noise_token if terminated else self._obs_sequence[self._step_count]

        return TimeStep(
            reward=np.array([reward]),
            terminated=np.array([terminated]),
            truncated=np.array([False]),
            observation=np.array([[obs]], dtype=np.int32),
            info={}
        )