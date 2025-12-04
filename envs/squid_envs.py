"""Custom squid environment creators for nash_ppo."""

from omegaconf import DictConfig
from envs.squid import CopyingEnv


def create_squid_env(env_config: DictConfig, **kwargs):
    """Create a custom squid environment.

    Args:
        env_config: Config with required fields:
            - squid_env_id: ID of the squid environment ("copying")
            - Additional environment-specific parameters
        **kwargs: Additional arguments (unused for squid envs)

    Returns:
        Squid environment instance (already implements BaseEnv interface)

    Raises:
        ValueError: If squid_env_id is not recognized or required params are missing
    """
    env_id = env_config.get("squid_env_id")
    if env_id is None:
        raise ValueError("env_config must contain 'squid_env_id' field for squid environments")

    if env_id == "copying":
        # Extract required parameters for CopyingEnv
        sequence_length = env_config.get("sequence_length")
        vocab_size = env_config.get("vocab_size")
        num_tokens_to_remember = env_config.get("num_tokens_to_remember")

        if sequence_length is None or vocab_size is None or num_tokens_to_remember is None:
            raise ValueError(
                "CopyingEnv requires 'sequence_length', 'vocab_size', and 'num_tokens_to_remember' "
                "in env_config"
            )

        # Optional parameters
        selective_copy = env_config.get("selective_copy", False)
        seed = env_config.get("seed", None)

        return CopyingEnv(
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            num_tokens_to_remember=num_tokens_to_remember,
            selective_copy=selective_copy,
            seed=seed
        )
    else:
        raise ValueError(
            f"Unknown squid environment ID: '{env_id}'. "
            f"Available: ['copying']"
        )


__all__ = ["create_squid_env"]
