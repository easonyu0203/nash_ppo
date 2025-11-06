"""
Training module for Nash PPO.

Organized into:
- algorithms/: PPO update logic and value normalization
- data/: Trajectory collection, buffering, and processing
- infrastructure/: Learner state, training loop, logging, types
"""

# Core algorithm components
from train.algorithms import update_agent, ValueNorm

# Data handling
from train.data import (
    RolloutBuffer,
    collect_trajectories,
    process_transitions,
)

# Infrastructure
from train.infrastructure import (
    LearnerState,
    create_learner_state,
    create_rollout_buffer,
    create_training_metrics,
    create_rollout_metrics,
    training_step,
    log_metrics,
    save_checkpoint,
    run_training_loop,
    create_logger,
    BaseLogger,
)

__all__ = [
    # Algorithms
    "update_agent",
    "ValueNorm",
    # Data
    "RolloutBuffer",
    "collect_trajectories",
    "process_transitions",
    # Infrastructure
    "LearnerState",
    "create_learner_state",
    "create_rollout_buffer",
    "create_training_metrics",
    "create_rollout_metrics",
    "training_step",
    "log_metrics",
    "save_checkpoint",
    "run_training_loop",
    "create_logger",
    "BaseLogger",
]
