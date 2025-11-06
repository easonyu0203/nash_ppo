"""Type definitions for agents."""

from enum import Enum


class ActionSpaceType(str, Enum):
    """Enumeration of supported action space types.

    Attributes:
        DISCRETE: Standard discrete action space (single integer action)
        MULTI_DISCRETE: Multiple discrete action spaces (tuple of integers)
        CONTINUOUS: Continuous action space (vector of floats)
    """
    DISCRETE = "discrete"
    MULTI_DISCRETE = "multi_discrete"
    CONTINUOUS = "continuous"
