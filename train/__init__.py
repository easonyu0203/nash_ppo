"""
Training module for Nash PPO.

This module provides:
- Core training utilities (data preparation, agent updates)
- Setup functions (learner state, buffers, metrics)
- Training loop orchestration
- Logging utilities
"""

from train.setup import (
    LearnerState,
    create_learner_state,
    create_rollout_buffer,
    create_training_metrics,
    create_rollout_metrics,
    process_rollout_metrics,
    create_value_norm_metrics,
)
from train.training_loop import (
    training_step,
    log_metrics,
    save_checkpoint,
    run_training_loop,
)

__all__ = [
    # Setup utilities
    "LearnerState",
    "create_learner_state",
    "create_rollout_buffer",
    "create_training_metrics",
    "create_rollout_metrics",
    "process_rollout_metrics",
    "create_value_norm_metrics",
    # Training loop utilities
    "training_step",
    "log_metrics",
    "save_checkpoint",
    "run_training_loop",
]
