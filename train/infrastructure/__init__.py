"""
Infrastructure components for training.

This module contains:
- LearnerState and setup utilities
- Training loop orchestration
- Logging utilities
- Type definitions
"""

from train.infrastructure.learner import (
    LearnerState,
    create_learner_state,
    create_rollout_buffer,
    create_training_metrics,
    create_rollout_metrics,
    process_rollout_metrics,
    create_value_norm_metrics,
)
from train.infrastructure.loop import (
    training_step,
    log_metrics,
    save_checkpoint,
    run_training_loop,
)
from train.infrastructure.loggers import (
    BaseLogger,
    TensorBoardLogger,
    JSONLogger,
    MultiLogger,
    create_logger,
)

__all__ = [
    # Learner utilities
    "LearnerState",
    "create_learner_state",
    "create_rollout_buffer",
    "create_training_metrics",
    "create_rollout_metrics",
    "process_rollout_metrics",
    "create_value_norm_metrics",
    # Training loop
    "training_step",
    "log_metrics",
    "save_checkpoint",
    "run_training_loop",
    # Logging
    "BaseLogger",
    "TensorBoardLogger",
    "JSONLogger",
    "MultiLogger",
    "create_logger",
]
