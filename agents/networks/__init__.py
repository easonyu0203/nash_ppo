"""Network components for agents - feature extractors and policy heads."""

# Feature extractors are now organized in feature_extractors/ subdirectory
# Policy heads remain in this directory

from agents.networks.feature_extractors import (
    FeatureExtractor,
    MLPFeatureExtractor,
    CNNFeatureExtractor,
    RecurrentFeatureExtractor,
    LSTMFeatureExtractor,
)
from agents.networks.policy_heads import (
    PolicyHead,
    DiscretePolicyHead,
    MultiDiscretePolicyHead,
    ContinuousPolicyHead,
    create_policy_head,
)

__all__ = [
    'FeatureExtractor',
    'MLPFeatureExtractor',
    'CNNFeatureExtractor',
    'RecurrentFeatureExtractor',
    'LSTMFeatureExtractor',
    'PolicyHead',
    'DiscretePolicyHead',
    'MultiDiscretePolicyHead',
    'ContinuousPolicyHead',
    'create_policy_head',
]
