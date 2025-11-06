"""
Training script for simultaneous update self-play (Refactored)

Usage:
    JAX_PLATFORMS=cpu uv run train/nash_pg_refactored.py \
                        algorithm.num_inner_update=1000 \
                        algorithm.num_outer_update=25 \
                        algorithm.mag_coef=0.2 \
                        logging.save_interval=1000 \
                        run_name=robot_warehouse/ippo/default_run

    CUDA_VISIBLE_DEVICES=0 uv run train/nash_pg_refactored.py \
                            algorithm.num_inner_update=200 \
                            algorithm.num_outer_update=100 \
                            logging.save_interval=2000 \

Assumption:
* Action space is Discrete, MultiDiscrete, or Box (continuous)
* Action space and Observation space are same for all agents
"""

import os
import logging
import warnings

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('orbax').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

import jax
import hydra
from omegaconf import DictConfig

from envs import create_env
from train.setup import create_learner_state, create_rollout_buffer
from train.training_loop import run_training_loop
from train.loggers import create_logger


def validate_config(config: DictConfig) -> None:
    """
    Validate training configuration parameters.

    Args:
        config: Training configuration

    Raises:
        ValueError: If configuration is invalid
    """
    if config.algorithm.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {config.algorithm.num_envs}")

    if config.algorithm.num_inner_update % config.logging.log_interval != 0:
        raise ValueError("log_interval must be divisible by num_inner_update")


def main(config: DictConfig) -> None:
    """
    Main training function.

    Args:
        config: Hydra configuration
    """
    key = jax.random.key(config.seed)

    # Initialize resources that need cleanup
    env = None
    logger = None

    try:
        # Validate configuration
        validate_config(config)

        # Setup environment
        env = create_env(config.env, num_env=config.algorithm.num_envs)
        init_timestep = env.reset(seed=config.seed)

        # Setup learner state (agent, optimizer, metrics, etc.)
        key, learner_key = jax.random.split(key)
        learner_state = create_learner_state(config, init_timestep, learner_key)

        # Create rollout buffer
        buffer = create_rollout_buffer(
            env=env,
            num_envs=config.algorithm.num_envs,
            num_steps=config.algorithm.num_steps
        )

        # Setup logger
        logger = create_logger(config)
        logger.log_config(config)

        # Run training loop
        learner_state = run_training_loop(
            learner_state=learner_state,
            env=env,
            buffer=buffer,
            logger=logger,
            config=config
        )

    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"\nTraining failed with error: {e}")
        raise  # Re-raise to preserve stack trace
    finally:
        # Always cleanup resources
        logging.info("Cleaning up resources...")

        if logger is not None:
            try:
                logger.close()
            except Exception as e:
                logging.warning(f"Failed to close logger: {e}")

        if env is not None:
            try:
                env.close()
            except Exception as e:
                logging.warning(f"Failed to close environment: {e}")

        logging.info("Cleanup complete")


@hydra.main(version_base=None, config_path="../conf/default", config_name="nash_pg")
def hydra_main(config: DictConfig) -> None:
    """Hydra entry point."""
    main(config)


if __name__ == '__main__':
    hydra_main()
