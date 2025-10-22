"""Unity ML-Agents environment wrapper for MARL framework."""

from typing import Any, Dict, List, Optional, Tuple
from functools import cached_property

import numpy as np
from gymnasium import Space
import gymnasium.spaces as gym_spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple, BehaviorSpec

from envs.mytypes import BaseEnv, TimeStep, Action


class UnityEnvWrapper(BaseEnv):
    """
    Wrapper for Unity ML-Agents environments.

    Assumptions:
    - All behavior names (agent types) have identical action/observation spaces
    - DecisionRequester settings are consistent across all agents
    - Uses area-based replication (num_areas) instead of multiple Unity instances
    - All agents step synchronously
    - Supports only discrete/multi-discrete action spaces
    - Uses default action mask (all actions valid)

    Individual Agent Termination:
    - Agents can terminate individually (not all at once)
    - The wrapper combines data from both DecisionSteps and TerminalSteps
    - Episode is considered "done" when ALL agents are in TerminalSteps
    - Terminal agents' observations are included in returned data
    - Actions can be set for terminal agents (Unity ignores them)
    """

    def __init__(
        self,
        file_name: Optional[str] = None,  # None = Editor, str = Built executable
        num_areas: int = 1,                # Number of parallel training areas
        base_port: Optional[int] = None,   # None = auto-select (5004 for editor, 5005 for builds)
        time_scale: float = 20.0,
        seed: int = 0,
        worker_id: int = 0,
        no_graphics: bool = True,
        additional_args: Optional[List[str]] = None,
    ):
        """
        Initialize Unity environment wrapper.

        Args:
            file_name: Path to Unity executable (None for Editor mode)
            num_areas: Number of parallel training areas in Unity
            base_port: Base port for communication
                      None (default): auto-select 5004 for editor, 5005 for builds
                      Actual port used = base_port + worker_id
            time_scale: Unity time scale (higher = faster training)
            seed: Random seed
            worker_id: Worker ID for port offset (port = base_port + worker_id)
            no_graphics: Disable graphics for faster training
            additional_args: Additional command line arguments for Unity
        """
        self._num_areas = num_areas
        self._time_scale = time_scale

        # Initialize side channels for configuration
        self._engine_channel = EngineConfigurationChannel()

        # Set engine configuration
        self._engine_channel.set_configuration_parameters(
            time_scale=time_scale,
            width=84,  # Default resolution for faster training
            height=84,
            quality_level=0,  # Lowest quality for speed
            target_frame_rate=-1,  # Unlimited
        )

        # UnityEnvironment handles base_port=None by auto-selecting:
        # - 5004 for editor (file_name=None)
        # - 5005 for builds (file_name specified)
        # Actual port = base_port + worker_id
        self._unity_env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            base_port=base_port,  # Pass None to let UnityEnvironment choose
            seed=seed,
            no_graphics=no_graphics,
            side_channels=[self._engine_channel],
            num_areas=num_areas,
            additional_args=additional_args,
        )
        self._unity_env.reset()

        # Validate environment meets framework requirements
        self._validate_environment()

        # mapping: agent_id (int) -> (area_idx, agent_idx)
        self._agent_id_to_idx: Dict[int, Tuple[int, int]] = {}
        # ordered list of agent ids in the mapping (flat order)
        self._agent_id_list: List[int] = []

        print(f"[UnityEnvWrapper] Initialized with {self._num_areas} areas, "
              f"{self._num_agents_per_area} agents per area, "
              f"time_scale={time_scale}")

    def _validate_environment(self):
        """Validate Unity environment meets framework requirements."""

        behavior_names = list(self._unity_env.behavior_specs.keys())

        # Check 1: At least one behavior exists
        if len(behavior_names) == 0:
            raise ValueError("No behaviors found in Unity environment")

        # Check 2: All behaviors have identical specs
        first_spec = self._unity_env.behavior_specs[behavior_names[0]]
        for name in behavior_names[1:]:
            spec = self._unity_env.behavior_specs[name]
            if not self._specs_equal(first_spec, spec):
                raise ValueError(
                    f"All behaviors must have identical action/observation spaces. "
                    f"Behavior '{name}' differs from '{behavior_names[0]}'"
                )

        self._behavior_names = behavior_names
        self._behavior_spec = first_spec # every behavior have same spec

        # Check 3: Only discrete or multi-discrete action spaces
        if first_spec.action_spec.continuous_size > 0:
            raise ValueError(
                f"Only discrete/multi-discrete actions supported. "
                f"Found {first_spec.action_spec.continuous_size} continuous actions"
            )

        if first_spec.action_spec.discrete_size == 0:
            raise ValueError("No discrete actions found in action spec")

        # Check 4: Validate num_agents matches across all areas
        self._validate_agent_count()

        print(f"[UnityEnvWrapper] Validation passed:")
        print(f"  - Behavior: {behavior_names}")
        print(f"  - Action space: {self.action_space}")
        print(f"  - Observation space: {self.observation_space}")

    def _validate_agent_count(self):
        """Ensure each area has same number of agents."""

        # Reset to get initial observations
        self._unity_env.reset()

        total_agents = 0
        for behavior_name in self._behavior_names:
            decision_steps, terminal_steps = self._unity_env.get_steps(behavior_name)

            if len(terminal_steps) != 0:
                raise ValueError("find termated agent after reset.")
            
            total_agents += len(decision_steps)

        if total_agents == 0:
            raise ValueError("No agents found after environment reset")

        if total_agents % self._num_areas != 0:
            raise ValueError(
                f"Total agents ({total_agents}) must be divisible by "
                f"num_areas ({self._num_areas}). "
                f"Ensure TrainingAreaReplicator created correct number of areas."
            )

        self._num_agents_per_area = total_agents // self._num_areas
        self._total_agents = total_agents

        print(f"  - Total agents: {total_agents} ({self._num_areas} areas × "
              f"{self._num_agents_per_area} agents)")

        # Pre-allocate buffers for observations, rewards, dones, action_masks
        # to avoid repeated allocations on every step
        self._allocate_buffers()

    def _specs_equal(self, spec1: BehaviorSpec, spec2: BehaviorSpec) -> bool:
        """Check if two behavior specs are equal."""

        # Compare action specs
        if (spec1.action_spec.continuous_size != spec2.action_spec.continuous_size or
            spec1.action_spec.discrete_size != spec2.action_spec.discrete_size):
            return False

        if not np.array_equal(
            spec1.action_spec.discrete_branches,
            spec2.action_spec.discrete_branches
        ):
            return False

        # Compare observation specs
        if len(spec1.observation_specs) != len(spec2.observation_specs):
            return False

        for obs1, obs2 in zip(spec1.observation_specs, spec2.observation_specs):
            if obs1.shape != obs2.shape:
                return False

        return True

    @cached_property
    def num_agents(self) -> int:
        """Number of agents per area."""
        return self._num_agents_per_area

    @cached_property
    def action_space(self) -> Space:
        """Convert ML-Agents ActionSpec to Gymnasium Space."""
        action_spec = self._behavior_spec.action_spec
        branches = action_spec.discrete_branches

        if action_spec.discrete_size == 1:
            # Single discrete action
            return gym_spaces.Discrete(int(branches[0]))
        else:
            # Multi-discrete action
            return gym_spaces.MultiDiscrete(np.array(branches).astype(np.int32))

    @cached_property
    def observation_space(self) -> Space:
        """Convert ML-Agents ObservationSpec to Gymnasium Box."""
        obs_specs = self._behavior_spec.observation_specs

        if len(obs_specs) > 1:
            raise ValueError("currently don't support multi-observation")
        
        return gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_specs[0].shape,
            dtype=np.float32
        )

    # ----------------- Helper methods -----------------
    def _allocate_buffers(self):
        """Pre-allocate reusable buffers to avoid repeated allocations on each step."""
        obs_shape = self._behavior_spec.observation_specs[0].shape
        self._obs_buffer = np.zeros(
            (self._num_areas, self._num_agents_per_area, *obs_shape),
            dtype=np.float32
        )
        self._reward_buffer = np.zeros(
            (self._num_areas, self._num_agents_per_area),
            dtype=np.float32
        )
        self._done_buffer = np.zeros(
            (self._num_areas, self._num_agents_per_area),
            dtype=bool
        )
        # Cache the default action mask (all actions valid)
        branches = self._behavior_spec.action_spec.discrete_branches
        if self._behavior_spec.action_spec.discrete_size == 1:
            n = int(branches[0])
            self._action_mask_buffer = np.ones(
                (self._num_areas, self._num_agents_per_area, n),
                dtype=bool
            )
        else:
            nvec = int(len(branches))
            self._action_mask_buffer = np.ones(
                (self._num_areas, self._num_agents_per_area, nvec),
                dtype=bool
            )

    def _build_agent_mapping_from_current_steps(self):
        """
        Build a deterministic mapping from Unity agent_id -> (area_idx, agent_idx)
        using current DecisionSteps ordering across behaviors.

        This must be called after a reset (when no terminal steps are present).
        """
        agent_map: Dict[int, Tuple[int, int]] = {}
        agent_list: List[int] = []

        # iterate behaviors in self._behavior_names order and their decision steps
        for behavior_name in self._behavior_names:
            decision_steps, terminal_steps = self._unity_env.get_steps(behavior_name)
            # Expect terminal_steps to be empty after reset
            for aid in decision_steps.agent_id:
                aid_int = int(aid)
                if aid_int in agent_map:
                    continue
                idx = len(agent_list)
                area_idx = idx // self._num_agents_per_area
                local_idx = idx % self._num_agents_per_area
                agent_map[aid_int] = (area_idx, local_idx)
                agent_list.append(aid_int)

        if len(agent_list) != self._total_agents:
            # If we didn't find expected number of agents, something is wrong.
            raise RuntimeError(
                f"Built agent mapping with {len(agent_list)} agents, expected {self._total_agents}"
            )

        self._agent_id_to_idx = agent_map
        self._agent_id_list = agent_list

    def _empty_batched_buffers(self):
        """
        Zero out and return pre-allocated buffers for observations, rewards, dones, action_mask.

        This avoids repeated memory allocations on every step by reusing buffers.
        """
        # Zero out the buffers
        self._obs_buffer.fill(0)
        self._reward_buffer.fill(0)
        self._done_buffer.fill(False)
        # Action mask buffer already contains all True values and doesn't need resetting
        # since it represents the default "all actions valid" state

        return self._obs_buffer, self._reward_buffer, self._done_buffer, self._action_mask_buffer

    def _collect_steps(self):
        """
        Collect current DecisionSteps and TerminalSteps from Unity and populate
        batched arrays using the agent_id -> (area, local_idx) mapping.
        """
        observations, rewards, dones, action_masks = self._empty_batched_buffers()

        # iterate behaviors and combine decision + terminal steps
        for behavior_name in self._behavior_names:
            decision_steps, terminal_steps = self._unity_env.get_steps(behavior_name)

            # DecisionSteps (not terminal)
            if len(decision_steps) > 0:
                # decision_steps.obs is a list (one element per observation); we assume single obs
                obs_arr = decision_steps.obs[0]
                rew_arr = decision_steps.reward
                # iterate in Unity-provided order
                for i, aid in enumerate(decision_steps.agent_id):
                    aid_int = int(aid)
                    if aid_int not in self._agent_id_to_idx:
                        # Should not happen if mapping built correctly
                        raise KeyError(f"Unknown agent_id {aid_int} encountered in DecisionSteps")
                    area_idx, local_idx = self._agent_id_to_idx[aid_int]
                    observations[area_idx, local_idx] = obs_arr[i]
                    rewards[area_idx, local_idx] = float(rew_arr[i])
                    dones[area_idx, local_idx] = False

            # TerminalSteps (agents that just terminated)
            if len(terminal_steps) > 0:
                obs_arr = terminal_steps.obs[0]
                rew_arr = terminal_steps.reward
                for i, aid in enumerate(terminal_steps.agent_id):
                    aid_int = int(aid)
                    if aid_int not in self._agent_id_to_idx:
                        raise KeyError(f"Unknown agent_id {aid_int} encountered in TerminalSteps")
                    area_idx, local_idx = self._agent_id_to_idx[aid_int]
                    observations[area_idx, local_idx] = obs_arr[i]
                    rewards[area_idx, local_idx] = float(rew_arr[i])
                    dones[area_idx, local_idx] = True

        return observations, rewards, dones, action_masks

    # ----------------- Public API -----------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> TimeStep:
        """Reset Unity environment."""

        # Perform Unity reset
        self._unity_env.reset()

        # After reset, build stable mapping from Unity agent ids -> (area, local_idx)
        self._build_agent_mapping_from_current_steps()

        # Collect initial state (should be only decision steps, no terminals)
        observations, rewards, dones, action_masks = self._collect_steps()

        return TimeStep(
            reward=rewards,
            done=dones,
            observation=observations,
            action_mask=action_masks,
            info={}
        )

    def step(self, action: Action) -> TimeStep:
        """Step Unity environment with actions.

        `action` is expected in batched shape (num_areas, num_agents_per_area, ...)
        where ... depends on action type:
          - single discrete: scalar per agent
          - multi-discrete: vector per agent with length == number of branches
        """
        action_spec = self._behavior_spec.action_spec
        is_single_discrete = (action_spec.discrete_size == 1)
        branches = action_spec.discrete_branches
        # For each behavior, get current DecisionSteps (agents that need actions)
        for behavior_name in self._behavior_names:
            decision_steps, terminal_steps = self._unity_env.get_steps(behavior_name)
            n_decision = len(decision_steps)
            if n_decision == 0:
                # nothing to set for this behavior
                continue

            # Build action array in Unity's decision order
            if is_single_discrete:
                # ActionTuple expects shape (n_agents, 1) for discrete actions
                act_arr = np.zeros((n_decision, 1), dtype=np.int32)
                for i, aid in enumerate(decision_steps.agent_id):
                    aid_int = int(aid)
                    if aid_int not in self._agent_id_to_idx:
                        raise KeyError(f"Unknown agent_id {aid_int} when preparing actions")
                    area_idx, local_idx = self._agent_id_to_idx[aid_int]
                    act_arr[i, 0] = action[area_idx, local_idx]
            else:
                # Multi-discrete: action per agent should be a vector of length = len(branches)
                n_branches = len(branches)
                act_arr = np.zeros((n_decision, n_branches), dtype=np.int32)
                for i, aid in enumerate(decision_steps.agent_id):
                    aid_int = int(aid)
                    if aid_int not in self._agent_id_to_idx:
                        raise KeyError(f"Unknown agent_id {aid_int} when preparing actions")
                    area_idx, local_idx = self._agent_id_to_idx[aid_int]
                    act_arr[i, :] = action[area_idx, local_idx]

            # Send actions for this behavior
            action_tuple = ActionTuple(discrete=act_arr)
            self._unity_env.set_actions(behavior_name, action_tuple)

        # Now advance Unity
        self._unity_env.step()

        # Collect the resulting steps (observations, rewards, dones)
        observations, rewards, dones, action_masks = self._collect_steps()

        return TimeStep(
            reward=rewards,
            done=dones,
            observation=observations,
            action_mask=action_masks,
            info={}
        )


    def close(self):
        """Close Unity environment."""
        self._unity_env.close()

    def __repr__(self) -> str:
        beh = self._behavior_names[0] if hasattr(self, "_behavior_names") and len(self._behavior_names) > 0 else "N/A"
        return (f"UnityEnvWrapper(behavior={beh}, "
                f"num_areas={self._num_areas}, "
                f"num_agents_per_area={self.num_agents})")
