"""
Training script for simultaneous update self-play

Usage:
    JAX_PLATFORMS=cpu uv run train/nash_pg.py \
                        algorithm.num_inner_update=1000 \
                        algorithm.num_outer_update=25 \
                        algorithm.mag_coef=0.2 \
                        logging.save_interval=1000 \
                        run_name=robot_warehouse/ippo/default_run

    CUDA_VISIBLE_DEVICES=0 uv run train/nash_pg.py \
                            algorithm.num_inner_update=200 \
                            algorithm.num_outer_update=100 \
                            logging.save_interval=2000 \

Assumption:
* Action space is Discrete, MultiDiscrete, or Box (continuous)
* Action space and Observation space are same for all agents
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
import logging
import warnings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Suppress verbose Orbax checkpoint logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('orbax').setLevel(logging.ERROR)

# Suppress FutureWarning from JAX scatter operations (from Jumanji library)
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax import nnx
import chex
import optax
import hydra
from omegaconf import DictConfig
from gymnasium.spaces import Dict as DictSpace, Discrete, MultiDiscrete, Box

from envs import create_env
import envs.mytypes as env_types
from agents import create_agent, BaseAgent
from train.core import update_agent, collect_trajectories, process_transitions, RolloutBuffer, ValueNorm
from train.loggers import create_logger, BaseLogger



@dataclass
class LearnerState:
    key: chex.PRNGKey
    last_timestep: env_types.TimeStep
    agent: BaseAgent
    optimizer: nnx.Optimizer
    train_metrics: nnx.MultiMetric
    rollout_metrics: nnx.MultiMetric
    mag_agent: Optional[BaseAgent] # use for regularization
    value_normalizer: Optional[ValueNorm] # use for value normalization


def training_step(
        learner_state: LearnerState,
        env: env_types.BaseEnv,
        config: DictConfig,
        buffer: RolloutBuffer
    ) -> LearnerState:
    """
    Single training step: collect trajectories and update agent.
    """
    
    """collect and process trajactories """
    learner_state.key, collect_key, update_key = jax.random.split(learner_state.key, 3)

    # Collect trajectories (num_envs, num_steps, num_agents)
    learner_state.last_timestep, transitions, next_value, next_terminated = collect_trajectories(
        env=env,
        agent=learner_state.agent,
        last_timestep=learner_state.last_timestep,
        key=collect_key,
        num_steps=config.algorithm.num_steps,
        buffer=buffer
    )

    learner_state.rollout_metrics, dataset = process_transitions(
        transitions, learner_state.rollout_metrics,
        next_value, next_terminated,
        gamma = config.algorithm.gamma,
        gae_gamma = config.algorithm.gae_gamma,
        value_normalizer = learner_state.value_normalizer
    )


    """perform ppo update"""
    learner_state.agent, learner_state.optimizer, learner_state.train_metrics = update_agent(
        agent = learner_state.agent,
        mag_agent = learner_state.mag_agent,
        optimizer = learner_state.optimizer,
        dataset = dataset,
        metrics = learner_state.train_metrics,
        key = update_key,
        ent_coef = config.algorithm.ent_coef,
        mag_coef = config.algorithm.mag_coef,
        clip_eps = config.algorithm.clip_eps,
        num_minibatches = config.algorithm.num_minibatches,
        num_ppo_epoch = config.algorithm.num_ppo_epoch,
        normalize_logprob = config.algorithm.normalize_logprob,
        value_normalizer = learner_state.value_normalizer,
    )

    return learner_state

def log_metrics(learner_state: LearnerState, logger: BaseLogger, cur_num_update: int):
    """Log training and rollout metrics"""

    train_metrics = learner_state.train_metrics.compute()
    rollout_metrics = learner_state.rollout_metrics.compute()

    # Log train metrics
    logger.log_train_metrics(train_metrics, cur_num_update)

    # Process and log rollout metrics
    eps_len = 1 / rollout_metrics['inverse_eps_len']
    ret = rollout_metrics['reward'] / rollout_metrics['inverse_eps_len']
    processed_rollout_metrics = {
        'eps_len': eps_len,
        'return': ret
    }
    logger.log_rollout_metrics(processed_rollout_metrics, cur_num_update)

    # Log value normalization statistics if enabled
    if learner_state.value_normalizer is not None:
        mean, var = learner_state.value_normalizer.running_mean_var()
        value_norm_metrics = {
            'value_norm/mean': mean[0],  # Extract scalar from shape (1,)
            'value_norm/std': jnp.sqrt(var)[0],
            'value_norm/debiasing_term': learner_state.value_normalizer.debiasing_term.value
        }
        logger.log_train_metrics(value_norm_metrics, cur_num_update)

    learner_state.train_metrics.reset()
    learner_state.rollout_metrics.reset()


def main(config: DictConfig):
    key = jax.random.key(config.seed)

    # Initialize resources that need cleanup
    env = None
    logger = None

    try:
        # Validate num_envs
        if config.algorithm.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {config.algorithm.num_envs}")

        # setup env
        env = create_env(config.env, num_env=config.algorithm.num_envs)
        init_timestep = env.reset(seed=config.seed)

        # setup agent
        key, agent_key = jax.random.split(key)
        agent = create_agent(config.agent, key=agent_key)

        # setup optimizer & metrics
        optimizer = nnx.Optimizer(agent, optax.adamw(config.algorithm.lr, eps=1e-5), wrt=nnx.Param)
        train_metrics = nnx.MultiMetric(
            actor_loss = nnx.metrics.Average("actor_loss"),
            ppo_loss = nnx.metrics.Average("ppo_loss"),
            entropy = nnx.metrics.Average("entropy"),
            critic_loss = nnx.metrics.Average("critic_loss"),
            approx_kl = nnx.metrics.Average("approx_kl"),
            mag_kl = nnx.metrics.Average("mag_kl"),
            clip_frac = nnx.metrics.Average("clip_frac"),
            explained_var = nnx.metrics.Average("explained_var"),
        )
        rollout_metrics = nnx.MultiMetric(
            inverse_eps_len = nnx.metrics.Average("inverse_eps_len"),
            reward = nnx.metrics.Average("reward"),
        )

        # setup value normalizer
        value_normalizer = None
        if config.algorithm.normalize_value:
            value_normalizer = ValueNorm()

        # setup learner state
        key, learner_key = jax.random.split(key)
        learner_state = LearnerState(
            key=learner_key,
            last_timestep=init_timestep,
            agent=agent,
            optimizer=optimizer,
            train_metrics=train_metrics,
            rollout_metrics=rollout_metrics,
            mag_agent=nnx.clone(agent), # init as the same
            value_normalizer=value_normalizer,
        )

        # create buffer for CPU-side rollout storage
        # Handle Dict observation spaces
        if isinstance(env.observation_space, DictSpace):
            obs_shape = {key: space.shape for key, space in env.observation_space.spaces.items()}
        else:
            obs_shape = env.observation_space.shape

        if isinstance(env.action_space, Discrete):
            action_shape = ()
        elif isinstance(env.action_space, MultiDiscrete):
            action_shape = env.action_space.nvec.shape
        elif isinstance(env.action_space, Box):
            # Continuous action space
            action_shape = env.action_space.shape
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

        buffer = RolloutBuffer(
            num_envs=config.algorithm.num_envs,
            num_steps=config.algorithm.num_steps,
            num_agents=env.num_agents,
            obs_shape=obs_shape,
            action_shape=action_shape
        )

        # setup logger
        logger = create_logger(config)
        logger.log_config(config)
        assert config.algorithm.num_inner_update % config.logging.log_interval == 0, "log_interval must be a divisible by num_update"

        # save first model
        if config.logging.save_interval > 0:
            learner_state.agent.save_checkpoint(Path(config.logging.checkpoint_dir).resolve() / config.run_name, step=0)

        # training loop
        with tqdm(total=config.algorithm.num_inner_update * config.algorithm.num_outer_update, desc="Training") as pbar:
            for cur_num_outer_update in range(0, config.algorithm.num_outer_update):
                for cur_num_inner_update in range(0, config.algorithm.num_inner_update, config.logging.log_interval):
                    cur_num_update = cur_num_outer_update * config.algorithm.num_inner_update + cur_num_inner_update

                    # training step for `log_interval` steps
                    for _ in range(config.logging.log_interval):
                        learner_state = training_step(learner_state, env=env, config=config, buffer=buffer)

                    # update progress bar
                    cur_num_update += config.logging.log_interval
                    pbar.update(config.logging.log_interval)

                    # logging
                    log_metrics(learner_state, logger, cur_num_update)

                    # save model
                    if config.logging.save_interval > 0 and cur_num_update % config.logging.save_interval == 0:
                        learner_state.agent.save_checkpoint(Path(config.logging.checkpoint_dir).resolve() / config.run_name, step=cur_num_update)

                # update magnet
                learner_state.mag_agent = nnx.clone(learner_state.agent)

    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    except Exception as e:
        logging.error(f"\nTraining failed with error: {e}")
        raise  # Re-raise to preserve stack trace
    finally:
        # CRITICAL: Always cleanup resources
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
    main(config)

if __name__ == '__main__':
    hydra_main()