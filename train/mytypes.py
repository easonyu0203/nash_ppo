import chex

import envs.mytypes as env_types

@chex.dataclass
class Transition:
    """
    Transition data collected during trajectory rollout.

    Contains raw environment interactions and agent outputs.
    Shape: (num_envs, num_steps, ...)

    Note: All agents act simultaneously each timestep.
    - action, value, log_prob: shape (num_agents,)
    - reward: shape (num_agents,)
    - observation: shape (num_agents, ...)
    """
    done: chex.Array      # bool, episode boundaries
    action: chex.Array          # agent actions, shape (num_agents,)
    value: chex.Array           # critic value estimates, shape (num_agents,)
    reward: chex.Array          # environment rewards per agent, shape (num_agents,)
    log_prob: chex.Array        # action log probabilities, shape (num_agents,)
    observation: env_types.Observation  # environment observations, shape (num_agents, ...)

@chex.dataclass
class Dataset:
    """
    Processed training dataset from transitions.

    Includes computed advantages and target values for training.
    Shape: (num_envs * num_steps * num_agents, ...)

    Note: Flattened across all agents for batch training.
    """
    action: chex.Array          # agent actions
    value: chex.Array           # critic value estimates
    log_prob: chex.Array        # action log probabilities
    observation: env_types.Observation  # environment observations
    advantage: chex.Array       # GAE advantages
    target_value: chex.Array    # critic training targets
    valid_mask: chex.Array      # bool mask: True if transition is valid for training
                                # False if state was terminal/truncated (invalid transition)
