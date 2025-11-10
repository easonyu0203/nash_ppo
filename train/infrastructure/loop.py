"""
Training loop utilities and orchestration.

This module contains functions for managing the training loop,
checkpoint saving, and metric logging.
"""

from pathlib import Path

import jax
from flax import nnx
from omegaconf import DictConfig
from tqdm import tqdm

import envs.mytypes as env_types
from train.data import collect_trajectories, process_transitions, RolloutBuffer
from train.infrastructure.loggers import BaseLogger
from train.infrastructure.learner import LearnerState, process_rollout_metrics, create_value_norm_metrics


def training_step(
    learner_state: LearnerState,
    env: env_types.BaseEnv,
    config: DictConfig,
    buffer: RolloutBuffer
) -> LearnerState:
    """
    Execute a single training step: collect trajectories and update agent.

    Args:
        learner_state: Current learner state
        env: Environment instance
        config: Training configuration
        buffer: Rollout buffer for trajectory storage

    Returns:
        Updated learner state
    """
    # Split random key
    learner_state.key, collect_key, update_key = jax.random.split(learner_state.key, 3)

    # Collect trajectories (num_envs, num_steps, num_agents)
    # Pass carries for stateful agents to maintain temporal continuity
    learner_state.last_timestep, transitions, next_value, next_terminated, learner_state.carries = collect_trajectories(
        env=env,
        agent=learner_state.agent,
        last_timestep=learner_state.last_timestep,
        key=collect_key,
        num_steps=config.algorithm.num_steps,
        buffer=buffer,
        carries=learner_state.carries
    )

    # Process transitions and compute advantages
    # Pass bptt_length for stateful agents
    from agents import StatefulAgent
    bptt_length = config.algorithm.bptt_length if isinstance(learner_state.agent, StatefulAgent) else None

    learner_state.rollout_metrics, dataset = process_transitions(
        transitions, learner_state.rollout_metrics,
        next_value, next_terminated,
        gamma=config.algorithm.gamma,
        gae_gamma=config.algorithm.gae_gamma,
        value_normalizer=learner_state.value_normalizer,
        bptt_length=bptt_length
    )

    # Perform PPO update (import here to avoid circular dependency)
    from train.algorithms import update_agent
    learner_state.agent, learner_state.optimizer, learner_state.train_metrics = update_agent(
        agent=learner_state.agent,
        mag_agent=learner_state.mag_agent,
        optimizer=learner_state.optimizer,
        dataset=dataset,
        metrics=learner_state.train_metrics,
        key=update_key,
        ent_coef=config.algorithm.ent_coef,
        mag_coef=config.algorithm.mag_coef,
        clip_eps=config.algorithm.clip_eps,
        num_minibatches=config.algorithm.num_minibatches,
        num_ppo_epoch=config.algorithm.num_ppo_epoch,
        normalize_logprob=config.algorithm.normalize_logprob,
        value_normalizer=learner_state.value_normalizer,
    )

    return learner_state


def log_metrics(
    learner_state: LearnerState,
    logger: BaseLogger,
    step: int
) -> None:
    """
    Log training and rollout metrics, then reset metric trackers.

    Args:
        learner_state: Current learner state with metrics
        logger: Logger instance
        step: Current training step number
    """
    # Compute metrics
    train_metrics = learner_state.train_metrics.compute()
    rollout_metrics = learner_state.rollout_metrics.compute()

    # Log train metrics
    logger.log_train_metrics(train_metrics, step)

    # Process and log rollout metrics
    processed_rollout_metrics = process_rollout_metrics(rollout_metrics)
    logger.log_rollout_metrics(processed_rollout_metrics, step)

    # Log value normalization statistics if enabled
    if learner_state.value_normalizer is not None:
        value_norm_metrics = create_value_norm_metrics(learner_state.value_normalizer)
        logger.log_train_metrics(value_norm_metrics, step)

    # Reset metrics
    learner_state.train_metrics.reset()
    learner_state.rollout_metrics.reset()


def save_checkpoint(
    learner_state: LearnerState,
    checkpoint_dir: str,
    run_name: str,
    step: int
) -> None:
    """
    Save agent checkpoint to disk.

    Args:
        learner_state: Current learner state
        checkpoint_dir: Base checkpoint directory
        run_name: Run name for organizing checkpoints
        step: Current training step
    """
    checkpoint_path = Path(checkpoint_dir).resolve() / run_name
    learner_state.agent.save_checkpoint(checkpoint_path, step=step)


def run_training_loop(
    learner_state: LearnerState,
    env: env_types.BaseEnv,
    buffer: RolloutBuffer,
    logger: BaseLogger,
    config: DictConfig
) -> LearnerState:
    """
    Execute the main training loop with logging and checkpointing.

    Args:
        learner_state: Initial learner state
        env: Environment instance
        buffer: Rollout buffer
        logger: Logger instance
        config: Training configuration

    Returns:
        Final learner state after training
    """
    # Save initial checkpoint
    if config.logging.save_interval > 0:
        save_checkpoint(learner_state, config.logging.checkpoint_dir, config.run_name, step=0)

    # Calculate total training steps
    total_steps = config.algorithm.num_inner_update * config.algorithm.num_outer_update

    with tqdm(total=total_steps, desc="Training") as pbar:
        for cur_outer_update in range(config.algorithm.num_outer_update):
            # Inner training loop
            for cur_inner_update in range(0, config.algorithm.num_inner_update, config.logging.log_interval):
                # Calculate global step
                global_step = cur_outer_update * config.algorithm.num_inner_update + cur_inner_update

                # Training steps for one log interval
                for _ in range(config.logging.log_interval):
                    learner_state = training_step(learner_state, env=env, config=config, buffer=buffer)

                # Update progress bar and step counter
                global_step += config.logging.log_interval
                pbar.update(config.logging.log_interval)

                # Log metrics
                log_metrics(learner_state, logger, global_step)

                # Save checkpoint
                if config.logging.save_interval > 0 and global_step % config.logging.save_interval == 0:
                    save_checkpoint(learner_state, config.logging.checkpoint_dir, config.run_name, global_step)

            # Update magnet agent at end of outer loop
            learner_state.mag_agent = nnx.clone(learner_state.agent)

    return learner_state
