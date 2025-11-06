"""
Data handling for reinforcement learning training.

This module contains:
- RolloutBuffer: CPU-side trajectory storage
- collect_trajectories: Environment interaction and data collection
- process_transitions: GAE calculation and dataset creation
"""

from train.data.buffer import RolloutBuffer
from train.data.collection import collect_trajectories
from train.data.processing import (
    process_transitions,
    calculate_gae,
    rearrange_transitions,
    create_dataset
)

__all__ = [
    "RolloutBuffer",
    "collect_trajectories",
    "process_transitions",
    "calculate_gae",
    "rearrange_transitions",
    "create_dataset",
]
