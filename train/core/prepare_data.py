from typing import Tuple, Optional
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import chex

import train.mytypes as train_types
import envs.mytypes as env_types
from agents import BaseAgent
from train.core.value_norm import ValueNorm


class RolloutBuffer:
    """
    CPU-side buffer for storing rollout data before transferring to GPU.
    Stores data as NumPy arrays for efficient CPU environment interaction.
    Supports both array and Dict observation spaces.
    """
    def __init__(self, num_envs: int, num_steps: int, num_agents: int, obs_shape, action_shape: tuple, action_mask_shape: tuple):
        """
        Args:
            num_envs: Number of parallel environments
            num_steps: Number of steps to collect
            num_agents: Number of agents per environment
            obs_shape: Shape of observations (per agent) - can be tuple for array obs or dict for Dict obs
            action_shape: Shape of actions (per agent)
                        - Discrete: () - scalar
                        - MultiDiscrete: (n,) - vector
            action_mask_shape: Shape of action masks (per agent)
                        - Discrete(n): (n,) - binary mask
                        - MultiDiscrete: (n,) - binary mask per dimension
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.step = 0

        # Preallocate buffers (NumPy arrays on CPU)
        self.dones = np.zeros((num_envs, num_steps, num_agents), dtype=np.bool_)
        self.actions = np.zeros((num_envs, num_steps, num_agents, *action_shape), dtype=np.int32)
        self.values = np.zeros((num_envs, num_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((num_envs, num_steps, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((num_envs, num_steps, num_agents), dtype=np.float32)

        # Support Dict observations
        if isinstance(obs_shape, dict):
            self.observations = {
                key: np.zeros((num_envs, num_steps, num_agents, *shape), dtype=np.float32)
                for key, shape in obs_shape.items()
            }
        else:
            self.observations = np.zeros((num_envs, num_steps, num_agents, *obs_shape), dtype=np.float32)

        self.action_masks = np.zeros((num_envs, num_steps, num_agents, *action_mask_shape), dtype=np.int8)

    def reset(self):
        """Reset buffer for new collection"""
        self.step = 0

    def add(self, done, action, value, reward, log_prob, observation, action_mask):
        """Add a timestep of data to buffer

        Args:
            done: Per-agent episode boundary flags, shape (num_envs, num_agents)
            action: Actions taken, shape (num_envs, num_agents, ...)
            value: Value estimates, shape (num_envs, num_agents)
            reward: Rewards received, shape (num_envs, num_agents)
            log_prob: Log probabilities, shape (num_envs, num_agents)
            observation: Observations, shape (num_envs, num_agents, ...) or dict of such
            action_mask: Action masks, shape (num_envs, num_agents, ...)
        """
        self.dones[:, self.step, :] = done
        self.actions[:, self.step] = action
        self.values[:, self.step] = value
        self.rewards[:, self.step] = reward
        self.log_probs[:, self.step] = log_prob

        # Handle Dict observations
        if isinstance(self.observations, dict):
            for key in self.observations.keys():
                self.observations[key][:, self.step] = observation[key]
        else:
            self.observations[:, self.step] = observation

        self.action_masks[:, self.step] = action_mask
        self.step += 1

    def to_jax(self) -> train_types.Transition:
        """Convert buffer to JAX arrays for GPU training"""
        return train_types.Transition(
            done=jnp.asarray(self.dones),
            action=jnp.asarray(self.actions),
            value=jnp.asarray(self.values),
            reward=jnp.asarray(self.rewards),
            log_prob=jnp.asarray(self.log_probs),
            observation=jax.tree.map(jnp.asarray, self.observations),
            action_mask=jnp.asarray(self.action_masks),
        )


def rearrange_transitions(transitions: train_types.Transition) -> train_types.Transition:
    """
    Rearrange transitions from (num_envs, num_steps, num_agents) to (num_envs, num_agents, num_steps).

    This makes the data natural for:
    - GAE calculation (batch of trajectories over time)
    - Flattening (num_envs * num_agents becomes batch dimension)
    """
    def swap_axes(x: chex.Array) -> chex.Array:
        """Swap step and agent dimensions: (num_envs, num_steps, num_agents, ...) -> (num_envs, num_agents, num_steps, ...)"""
        return jnp.swapaxes(x, 1, 2)

    return train_types.Transition(
        done=swap_axes(transitions.done),  # (num_envs, num_steps, num_agents) -> (num_envs, num_agents, num_steps)
        action=swap_axes(transitions.action),
        value=swap_axes(transitions.value),
        reward=swap_axes(transitions.reward),
        log_prob=swap_axes(transitions.log_prob),
        observation=jax.tree.map(swap_axes, transitions.observation),
        action_mask=swap_axes(transitions.action_mask),
    )


def create_dataset(
    transitions: train_types.Transition,
    advantages: chex.Array,
    target_values: chex.Array,
    batch_size: int
) -> train_types.Dataset:
    """
    Create flattened dataset from rearranged transitions.

    Args:
        transitions: Rearranged transitions with shape (num_envs, num_agents, num_steps, ...)
        advantages: Shape (num_envs, num_agents, num_steps)
        target_values: Shape (num_envs, num_agents, num_steps)
        batch_size: num_envs * num_agents * num_steps

    Returns:
        Dataset with all fields flattened to (batch_size, ...)
    """
    def flatten_to_batch(x: chex.Array) -> chex.Array:
        """Flatten (num_envs, num_agents, num_steps, ...) -> (batch_size, ...)"""
        return x.reshape(batch_size, *x.shape[3:])

    # Compute validity mask: True if transition is valid (state was NOT done)
    # When done=True, the action was taken in a terminal/truncated state and should be ignore
    valid_mask = ~transitions.done  # Shape: (num_envs, num_agents, num_steps)

    return train_types.Dataset(
        action=flatten_to_batch(transitions.action),
        value=transitions.value.reshape(batch_size),
        log_prob=transitions.log_prob.reshape(batch_size),
        observation=jax.tree.map(flatten_to_batch, transitions.observation),
        action_mask=flatten_to_batch(transitions.action_mask),
        advantage=advantages.reshape(batch_size),
        target_value=target_values.reshape(batch_size),
        valid_mask=valid_mask.reshape(batch_size),
    )


def collect_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_steps: int,
    buffer: RolloutBuffer
) -> Tuple[env_types.TimeStep, train_types.Transition, chex.Array, chex.Array]:
    """
    Collect trajectories from vectorized environments running on CPU.

    Data flow:
    1. Env step on CPU (NumPy) → store in buffer
    2. Agent inference on GPU (JAX)
    3. Convert buffer to JAX at end for training

    Args:
        env: Vectorized environment (e.g., DummyVecEnv)
             Returns data with shape (num_envs, num_agents, ...)
        agent: Shared agent model (JAX-based, runs on GPU)
        last_timestep: Last timestep from previous rollout (NumPy arrays)
                      Fields have shape (num_envs, num_agents, ...)
        key: JAX random key for action sampling
        num_steps: Number of steps to collect
        buffer: Preallocated RolloutBuffer for CPU-side storage

    Returns:
        Tuple containing:
        - last_timestep: Final timestep (NumPy arrays)
        - transitions: Collected transitions as JAX arrays for training
                      Shape: (num_envs, num_steps, num_agents, ...)
        - next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        - next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                          Used to zero out bootstrap when episode naturally ended

    Note:
        - Environments auto-reset when all agents are done
        - NumPy ↔ JAX conversions happen at agent inference boundary
    """
    # Get shapes from last_timestep
    # Handle Dict observations - get shape from first key
    if isinstance(last_timestep.observation, dict):
        first_key = next(iter(last_timestep.observation.keys()))
        num_envs = last_timestep.observation[first_key].shape[0]
        num_agents = last_timestep.observation[first_key].shape[1]
    else:
        num_envs = last_timestep.observation.shape[0]
        num_agents = last_timestep.observation.shape[1]

    buffer.reset()

    # Rollout loop (Python loop, can't JIT through NumPy env)
    for _ in range(num_steps):
        key, act_key = jax.random.split(key)

        # === Convert NumPy → JAX for agent inference ===
        obs_jax = jax.tree.map(jnp.asarray, last_timestep.observation)
        action_mask_jax = jnp.asarray(last_timestep.action_mask)

        # Flatten env and agent dims for agent: (num_envs, num_agents, ...) → (batch, ...)
        batch_size = num_envs * num_agents
        def flatten_batch(x):
            return x.reshape(batch_size, *x.shape[2:])
        obs_flat = jax.tree.map(flatten_batch, obs_jax)
        action_mask_flat = action_mask_jax.reshape(batch_size, *action_mask_jax.shape[2:])

        # Agent forward pass (JAX, runs on GPU)
        actions_flat, log_probs_flat, values_flat = agent.get_action_and_value(
            obs_flat, act_key, action_mask_flat
        )

        # Unflatten back: (batch,) → (num_envs, num_agents, ...)
        actions_jax = actions_flat.reshape(num_envs, num_agents, *actions_flat.shape[1:])
        log_probs_jax = log_probs_flat.reshape(num_envs, num_agents)
        values_jax = values_flat.reshape(num_envs, num_agents)

        # === Convert JAX → NumPy for env step ===
        actions_np = np.asarray(actions_jax)
        values_np = np.asarray(values_jax)
        log_probs_np = np.asarray(log_probs_jax)

        # Step environment (NumPy, runs on CPU)
        new_timestep = env.step(actions_np)

        # Compute done flag as terminated OR truncated for episode boundary tracking
        done_np = last_timestep.terminated | last_timestep.truncated

        # Store in buffer (all NumPy)
        buffer.add(
            done=done_np,              # (num_envs, num_agents) - per-agent episode boundaries
            action=actions_np,               # (num_envs, num_agents, ...)
            value=values_np,                 # (num_envs, num_agents)
            reward=new_timestep.reward,      # (num_envs, num_agents)
            log_prob=log_probs_np,           # (num_envs, num_agents)
            observation=last_timestep.observation,  # (num_envs, num_agents, ...)
            action_mask=last_timestep.action_mask,  # (num_envs, num_agents, ...)
        )

        last_timestep = new_timestep

    # Compute bootstrap value for GAE (value of the next state after rollout)
    obs_jax = jax.tree.map(jnp.asarray, last_timestep.observation)
    batch_size = num_envs * num_agents
    def flatten_batch_bootstrap(x):
        return x.reshape(batch_size, *x.shape[2:])
    obs_flat = jax.tree.map(flatten_batch_bootstrap, obs_jax)

    # Get value estimate for bootstrap
    next_value_flat = agent.get_value(obs_flat)
    next_value = next_value_flat.reshape(num_envs, num_agents)
    # Only use terminated for bootstrap - if truncated, we still want to bootstrap
    next_terminated = jnp.asarray(last_timestep.terminated)  # (num_envs, num_agents)

    # Convert buffer to JAX arrays for GPU training
    transitions = buffer.to_jax()

    return last_timestep, transitions, next_value, next_terminated

@nnx.jit
def process_transitions(
    transitions: train_types.Transition,
    metrics: nnx.MultiMetric,
    next_value: chex.Array,
    next_terminated: chex.Array,
    gamma: float,
    gae_gamma: float,
    value_normalizer: Optional[ValueNorm] = None,
) -> Tuple[nnx.MultiMetric, train_types.Dataset]:
    """
    Process transitions and calculate advantages using GAE.

    Args:
        transitions: Collected transitions with shape (num_envs, num_steps, num_agents, ...)
        metrics: Metric tracker for logging
        next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                        Used to zero out bootstrap when episode naturally ended (not truncated)
        gamma: Discount factor
        gae_gamma: GAE lambda parameter
        value_normalizer: Optional value normalizer for denormalizing value predictions

    Returns:
        Tuple of (updated metrics, training dataset)
    """
    # Step 2: Rearrange to (num_envs, num_agents, num_steps) for processing
    transitions = rearrange_transitions(transitions)
    num_envs, num_agents, num_steps = transitions.reward.shape

    # Step 3: Calculate GAE - works on (num_envs, num_agents, num_steps)
    advantages, target_values = calculate_gae(
        transitions, next_value, next_terminated, gamma, gae_gamma, value_normalizer
    )

    # Step 4: Flatten to batch dimension (num_envs * num_agents * num_steps,)
    batch_size = num_envs * num_agents * num_steps

    dataset = create_dataset(
        transitions, advantages, target_values, batch_size
    )

    # Log metrics - agent 0 reward only
    ag0_reward = transitions.reward[:, 0, :]  # (num_envs, num_steps)
    ag0_dones = transitions.done[:, 0, :]  # (num_envs, num_steps)
    metrics.update(
        inverse_eps_len=ag0_dones.reshape(num_envs * num_steps),
        reward=ag0_reward.reshape(num_envs * num_steps)
    )

    return metrics, dataset

@nnx.jit
def calculate_gae(
    transitions: train_types.Transition,
    next_value: chex.Array,
    next_terminated: chex.Array,
    gamma: float,
    gae_gamma: float,
    value_normalizer: Optional[ValueNorm] = None,
) -> Tuple[chex.Array, chex.Array]:
    """
    Calculate Generalized Advantage Estimation (GAE) for multi-agent trajectories.

    Strategy: Work directly on (num_envs, num_agents, num_steps) shape - flatten once and vmap.

    Args:
        transitions: trajectory data with shape (num_envs, num_agents, num_steps, ...)
                    - reward: (num_envs, num_agents, num_steps)
                    - value: (num_envs, num_agents, num_steps)
                    - done: (num_envs, num_agents, num_steps) - per-agent episode boundaries
        next_value: Bootstrap value for GAE, shape (num_envs, num_agents)
        next_terminated: Terminated flag for bootstrap, shape (num_envs, num_agents)
                        When True, episode naturally ended → don't bootstrap (use 0)
                        When False, episode was truncated → DO bootstrap (use next_value)
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        gae_gamma: GAE lambda parameter (0 < gae_gamma <= 1)
        value_normalizer: Optional value normalizer for denormalizing value predictions

    Returns:
        Tuple of (advantages, target_values) with shape (num_envs, num_agents, num_steps)
    """

    def calculate_single_trajectory_gae(
        trajectory_data: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Calculate GAE for a single trajectory (one env, one agent).

        Args:
            trajectory_data: Tuple of (rewards, values, dones, next_value, next_terminated)
                           - rewards, values, dones: shape (num_steps,)
                           - next_value: scalar
                           - next_terminated: scalar - True if naturally ended, False if truncated

        Returns:
            Tuple of (advantages, target_values), each shape (num_steps,)
        """
        rewards, values, dones, next_value, next_terminated = trajectory_data

        def gae_step(
            carry: Tuple[chex.Array, chex.Array],
            timestep_data: Tuple[chex.Array, chex.Array, chex.Array]
        ) -> Tuple[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]:
            """Single GAE step in reverse scan over time."""
            next_gae, next_value = carry
            reward, value, done = timestep_data

            # Compute TD error and GAE
            td_error = reward + gamma * next_value - value
            gae = td_error + gamma * gae_gamma * next_gae

            advantage = gae
            target_value = advantage + value

            # Reset carry at episode boundaries (prevent leakage to previous episode)
            gae_out = jnp.where(done, 0.0, gae)
            value_out = jnp.where(done, 0.0, value)

            return (gae_out, value_out), (advantage, target_value)

        # - If terminated (natural end): use 0 (no future value)
        # - If truncated (time limit): use next_value (episode continues, we just stopped collecting)
        bootstrap_value = jnp.where(next_terminated, 0.0, next_value)
        init_carry = (jnp.float32(0.0), bootstrap_value)

        _, (advantages, target_values) = jax.lax.scan(
            gae_step,
            init_carry,
            (rewards, values, dones),
            reverse=True
        )

        return advantages, target_values

    # Extract data - already in shape (num_envs, num_agents, num_steps)
    rewards = transitions.reward
    values = transitions.value
    dones = transitions.done  # Already (num_envs, num_agents, num_steps)

    # Denormalize values if using value normalization
    # Values from the network are in normalized space, need to convert to original scale for GAE
    if value_normalizer is not None:
        # Flatten for denormalization
        num_envs, num_agents, num_steps = values.shape
        values_flat = values.reshape(-1)
        next_value_flat = next_value.reshape(-1)

        # Denormalize
        values_denorm_flat = value_normalizer.denormalize(values_flat)
        next_value_denorm_flat = value_normalizer.denormalize(next_value_flat)

        # Reshape back
        values = values_denorm_flat.reshape(num_envs, num_agents, num_steps)
        next_value = next_value_denorm_flat.reshape(num_envs, num_agents)

    # Infer shapes from arrays
    num_envs, num_agents, num_steps = rewards.shape

    # Flatten to batch: (num_envs, num_agents, ...) → (num_envs * num_agents, ...)
    batch_size = num_envs * num_agents
    rewards_flat = rewards.reshape(batch_size, num_steps)
    values_flat = values.reshape(batch_size, num_steps)
    dones_flat = dones.reshape(batch_size, num_steps)
    next_value_flat = next_value.reshape(batch_size)
    next_terminated_flat = next_terminated.reshape(batch_size)

    # Vmap over batch dimension
    batched_gae = jax.vmap(calculate_single_trajectory_gae, in_axes=0, out_axes=0)
    advantages_flat, target_values_flat = batched_gae((
        rewards_flat, values_flat, dones_flat, next_value_flat, next_terminated_flat
    ))

    # Unflatten: (num_envs * num_agents, num_steps) → (num_envs, num_agents, num_steps)
    advantages = advantages_flat.reshape(num_envs, num_agents, num_steps)
    target_values = target_values_flat.reshape(num_envs, num_agents, num_steps)

    return advantages, target_values
