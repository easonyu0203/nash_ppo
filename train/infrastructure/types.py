from typing import Any, Optional
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
    - initial_carry: hidden state at start of timestep (for recurrent agents)
                     shape (num_agents, *carry_shape), None for stateless agents
    """
    done: chex.Array      # bool, episode boundaries
    action: chex.Array          # agent actions, shape (num_agents,)
    value: chex.Array           # critic value estimates, shape (num_agents,)
    reward: chex.Array          # environment rewards per agent, shape (num_agents,)
    log_prob: chex.Array        # action log probabilities, shape (num_agents,)
    observation: env_types.Observation  # environment observations, shape (num_agents, ...)
    initial_carry: Optional[Any] = None  # hidden state at start of timestep (for stateful agents)

@chex.dataclass
class Dataset:
    """
    Processed training dataset from transitions.

    For stateless agents:
        Shape: (batch_size, ...) where batch_size = num_envs * num_agents * num_steps
        Each sample is a single timestep

    For stateful agents:
        Shape: (batch_size, bptt_length, ...) where batch_size = num_envs * num_agents * num_steps // bptt_length
        Each sample is a sequence of bptt_length timesteps for BPTT training
    """
    action: chex.Array          # agent actions
    value: chex.Array           # critic value estimates
    log_prob: chex.Array        # action log probabilities
    observation: env_types.Observation  # environment observations
    advantage: chex.Array       # GAE advantages
    target_value: chex.Array    # critic training targets
    done: chex.Array            # bool mask: True if episode terminated/truncated at this step
    initial_carry: Optional[Any] = None  # initial hidden states at start of each BPTT segment (stateful agents only)
                                         # Shape: (batch_size, *carry_shape) where batch_size = num_envs * num_agents * num_steps // bptt_length
