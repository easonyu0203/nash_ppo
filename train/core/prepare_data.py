from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import chex

import train.mytypes as train_types
import envs.mytypes as env_types
from agents import BaseAgent


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
        is_new_eps=transitions.is_new_eps,  # (num_envs, num_steps) - no change
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

    return train_types.Dataset(
        action=transitions.action.reshape(batch_size),
        value=transitions.value.reshape(batch_size),
        log_prob=transitions.log_prob.reshape(batch_size),
        observation=jax.tree.map(flatten_to_batch, transitions.observation),
        action_mask=flatten_to_batch(transitions.action_mask),
        advantage=advantages.reshape(batch_size),
        target_value=target_values.reshape(batch_size),
    )


@partial(nnx.jit, static_argnames=('env', 'num_envs', 'num_steps'))
def collect_and_process_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    env_state: env_types.EnvState,
    last_timestep: env_types.TimeStep,
    metrics: nnx.MultiMetric,
    key: chex.PRNGKey,
    num_envs: int,
    num_steps: int,
    gamma: float,
    gae_gamma: float,
) -> Tuple[env_types.EnvState, env_types.TimeStep, nnx.MultiMetric, train_types.Dataset]:
    """
    Collect trajectories and process them into a training dataset.

    Data flow:
    1. collect_trajectories → (num_envs, num_steps, num_agents)
    2. Rearrange to (num_envs, num_agents, num_steps) - natural for GAE and flattening
    3. calculate_gae → (num_envs, num_agents, num_steps)
    4. Flatten to (num_envs * num_agents * num_steps,) for training

    Args:
        env: Environment instance
        agent: Shared agent model used by all agents
        env_state: Current environment state with shape (num_envs, ...)
        last_timestep: Last timestep from previous rollout with shape (num_envs, ...)
        metrics: Metric tracker for logging
        key: JAX random key (scalar)
        num_envs: Number of parallel environments
        num_steps: Number of steps to collect
        gamma: Discount factor for GAE (0 < gamma <= 1)
        gae_gamma: GAE lambda parameter for bias-variance tradeoff (0 < gae_gamma <= 1)

    Returns:
        Tuple of (new_env_state, new_last_timestep, new_metrics, dataset)
    """
    num_agents = env.num_agents

    # Step 1: Collect trajectories (num_envs, num_steps, num_agents)
    env_state, last_timestep, transitions = collect_trajectories(
        env, agent, env_state, last_timestep, key, num_envs, num_steps
    )

    # Step 2: Rearrange to (num_envs, num_agents, num_steps) for processing
    transitions = rearrange_transitions(transitions)

    # Step 3: Calculate GAE - works on (num_envs, num_agents, num_steps)
    advantages, target_values = calculate_gae(
        transitions, gamma, gae_gamma
    )

    # Step 4: Flatten to batch dimension (num_envs * num_agents * num_steps,)
    batch_size = num_envs * num_agents * num_steps

    dataset = create_dataset(
        transitions, advantages, target_values, batch_size
    )

    # Log metrics - mean reward across agents
    mean_reward = transitions.reward.mean(axis=1)  # (num_envs, num_steps)
    metrics.update(
        inverse_eps_len=transitions.is_new_eps.reshape(num_envs * num_steps),
        reward=mean_reward.reshape(num_envs * num_steps)
    )

    return env_state, last_timestep, metrics, dataset

@partial(nnx.jit, static_argnames=('env', 'num_envs', 'num_steps'))
def collect_trajectories(
    env: env_types.BaseEnv,
    agent: BaseAgent,
    env_state: env_types.EnvState,
    last_timestep: env_types.TimeStep,
    key: chex.PRNGKey,
    num_envs: int,
    num_steps: int,
) -> Tuple[env_types.EnvState, env_types.TimeStep, train_types.Transition]:
    """
    Collect trajectories from multiple environments for a specified number of steps.

    All agents act simultaneously using a shared agent model.

    Args:
        env: The environment instance implementing BaseEnv interface
        agent: Shared agent model used by all agents
        env_state: Current state of all environments, shape (num_envs, ...)
        last_timestep: The last timestep from previous rollout, shape (num_envs, ...)
        key: JAX random key for action sampling (single key, will be split internally)
        num_envs: Number of parallel environments to run
        num_steps: Number of steps to collect from each environment

    Returns:
        Tuple containing:
        - env_state: Updated environment states after rollout, shape (num_envs, ...)
        - last_timestep: Final timestep from rollout, shape (num_envs, ...)
        - transitions: Collected transitions, shape (num_envs, num_steps, ...)
                      All agent actions/values/log_probs have shape (num_agents,) per timestep

    Note:
        - Environments are assumed to auto-reset when done=True
        - All inputs must have consistent batch dimensions of num_envs
    """

    chex.assert_rank(key, 0) # one key

    def collect_one_env_step(carry: Tuple[env_types.TimeStep, env_types.EnvState, chex.PRNGKey], _: Any):
        """Step env for a single env, i.e., no batch dimensions. All agents act simultaneously."""
        last_timestep, env_state, key = carry
        key, act_key = jax.random.split(key)

        # Shared agent - use num_agents as batch dimension
        # observations shape: (num_agents, ...) → agent treats this as (batch_size, ...)
        actions, log_probs, values = agent.get_action_and_value(
            last_timestep.observation, act_key, last_timestep.action_mask
        )
        # Output shapes: actions, log_probs, values all (num_agents,)

        # Step env with actions from all agents (actions shape: (num_agents,))
        env_state, new_timestep = env.step(env_state, actions)

        # Create transition storing all agents' data
        transition = train_types.Transition(
            is_new_eps=last_timestep.done,  # our env auto reset, so when last step is done, this step is new episode
            action=actions,                  # shape (num_agents,)
            value=values,                    # shape (num_agents,)
            reward=new_timestep.reward,      # shape (num_agents,)
            log_prob=log_probs,              # shape (num_agents,)
            observation=last_timestep.observation,     # shape (num_agents, ...)
            action_mask=last_timestep.action_mask,     # shape (num_agents, ...)
        )

        return (new_timestep, env_state, key), transition

    # the output of rollout will have extra dimension of (num_steps, ...)
    single_env_rollout_fc = nnx.scan(
        collect_one_env_step, length=num_steps
    )

    # batch for num_envs
    batch_env_rollout_fc = nnx.vmap(single_env_rollout_fc, in_axes=(0, None), out_axes=0)

    # prepare batched keys
    keys = jax.random.split(key, num_envs)

    # perform batch rollout, return with shape (num_envs, num_steps, ...)
    (last_timestep, env_state, _), transitions = batch_env_rollout_fc(
        (last_timestep, env_state, keys), # carry
        None # empty Ys
    )

    chex.assert_shape(transitions.is_new_eps, (num_envs, num_steps))

    return env_state, last_timestep, transitions


@nnx.jit
def calculate_gae(
    transitions: train_types.Transition,
    gamma: float,
    gae_gamma: float,
) -> Tuple[chex.Array, chex.Array]:
    """
    Calculate Generalized Advantage Estimation (GAE) for multi-agent trajectories.

    Strategy: Work directly on (num_envs, num_agents, num_steps) shape - flatten once and vmap.

    Args:
        transitions: Rearranged trajectory data with shape (num_envs, num_agents, num_steps, ...)
                    - reward: (num_envs, num_agents, num_steps)
                    - value: (num_envs, num_agents, num_steps)
                    - is_new_eps: (num_envs, num_steps) - scalar per env/timestep
        gamma: Discount factor for future rewards (0 < gamma <= 1)
        gae_gamma: GAE lambda parameter (0 < gae_gamma <= 1)

    Returns:
        Tuple of (advantages, target_values) with shape (num_envs, num_agents, num_steps)
    """

    def calculate_single_trajectory_gae(
        trajectory_data: Tuple[chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Calculate GAE for a single trajectory (one env, one agent).

        Args:
            trajectory_data: Tuple of (rewards, values, is_new_eps), each shape (num_steps,)

        Returns:
            Tuple of (advantages, target_values), each shape (num_steps,)
        """
        rewards, values, is_new_eps = trajectory_data

        def gae_step(
            carry: Tuple[chex.Array, chex.Array],
            timestep_data: Tuple[chex.Array, chex.Array, chex.Array]
        ) -> Tuple[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]:
            """Single GAE step in reverse scan over time."""
            next_gae, next_value = carry
            reward, value, is_new_eps = timestep_data

            # Compute TD error and GAE
            td_error = reward + gamma * next_value - value
            gae = td_error + gamma * gae_gamma * next_gae

            advantage = gae
            target_value = advantage + value

            # Reset carry at episode boundaries (prevent leakage to previous episode)
            gae_out = jnp.where(is_new_eps, 0.0, gae)
            value_out = jnp.where(is_new_eps, 0.0, value)

            return (gae_out, value_out), (advantage, target_value)

        # Initialize and scan
        init_carry = (jnp.float32(0.0), jnp.float32(0.0))
        _, (advantages, target_values) = jax.lax.scan(
            gae_step,
            init_carry,
            (rewards, values, is_new_eps),
            reverse=True
        )

        return advantages, target_values

    # Extract data - already in shape (num_envs, num_agents, num_steps)
    rewards = transitions.reward
    values = transitions.value

    # Infer shapes from arrays
    num_envs, num_agents, num_steps = rewards.shape

    # Broadcast is_new_eps: (num_envs, num_steps) → (num_envs, num_agents, num_steps)
    is_new_eps = transitions.is_new_eps[:, None, :]  # (num_envs, 1, num_steps)
    is_new_eps = jnp.broadcast_to(is_new_eps, rewards.shape)

    # Flatten to batch: (num_envs, num_agents, num_steps) → (num_envs * num_agents, num_steps)
    batch_size = num_envs * num_agents
    rewards_flat = rewards.reshape(batch_size, num_steps)
    values_flat = values.reshape(batch_size, num_steps)
    is_new_eps_flat = is_new_eps.reshape(batch_size, num_steps)

    # Vmap over batch dimension
    batched_gae = jax.vmap(calculate_single_trajectory_gae, in_axes=0, out_axes=0)
    advantages_flat, target_values_flat = batched_gae((rewards_flat, values_flat, is_new_eps_flat))

    # Unflatten: (num_envs * num_agents, num_steps) → (num_envs, num_agents, num_steps)
    advantages = advantages_flat.reshape(num_envs, num_agents, num_steps)
    target_values = target_values_flat.reshape(num_envs, num_agents, num_steps)

    return advantages, target_values
