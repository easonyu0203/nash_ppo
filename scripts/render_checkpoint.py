"""
Render one episode using a trained agent checkpoint.

Usage:
    uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/robot_warehouse/nash_pg/default_run --step 2000 --env-config conf/env/robot_warehouse/tiny_2ag.yaml --seed 42
"""

import os
import logging
import warnings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cpu"
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('orbax').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Sharding info not provided.*")
import argparse
from functools import partial
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from envs import create_env
from agents import BaseAgent


@partial(jax.jit, static_argnames=('env', 'agent'))
def run_episode(env, agent, env_state, timestep, key):
    """JIT-compiled episode rollout for speed."""
    def step_fn(carry, _):
        env_state, timestep, key, episode_reward, done = carry

        # Get action
        key, action_key = jax.random.split(key)
        action = agent.get_action(timestep.observation, action_key, timestep.action_mask)

        # Step environment
        env_state, next_timestep = env.step(env_state, action)

        # Accumulate reward (only if not done)
        episode_reward = episode_reward + timestep.reward * (1 - done)

        return (env_state, next_timestep, key, episode_reward, next_timestep.done), (env_state, timestep)

    # Run until done (max 1000 steps to prevent infinite loop)
    init_carry = (env_state, timestep, key, jnp.zeros(env.num_agents), timestep.done)
    final_carry, (states, timesteps) = jax.lax.scan(step_fn, init_carry, None, length=1000)

    final_env_state, _, _, episode_reward, _ = final_carry

    # Find actual episode length (first done=True)
    dones = timesteps.done
    episode_length = jnp.argmax(dones) + 1  # +1 because argmax is 0-indexed

    return final_env_state, episode_reward, episode_length, states, timesteps


def play_episode(checkpoint_dir: str, step: int, env_config_path: str, seed: int = 0):
    """
    Load a checkpoint and play one episode with rendering.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        step: Training step to load
        env_config_path: Path to environment config yaml
        seed: Random seed for episode
    """
    # Load agent from checkpoint
    print(f"Loading checkpoint from {checkpoint_dir} at step {step}...")
    key = jax.random.key(seed)
    agent = BaseAgent.load_checkpoint(checkpoint_dir, step, key)
    print(f"Loaded agent: {agent.get_class_name()}")

    # Load environment config
    env_config = OmegaConf.load(env_config_path)
    print(f"Environment config: {OmegaConf.to_yaml(env_config)}")

    # Create environment
    env = create_env(env_config)
    print(f"Environment: {env}")
    print(f"Num agents: {env.num_agents}")

    # Reset environment
    key, reset_key = jax.random.split(key)
    env_state, timestep = env.reset(reset_key)

    # Run episode (JIT-compiled for speed)
    print("\nRunning episode (JIT-compiled)...")
    final_env_state, episode_reward, episode_length, states, _ = run_episode(
        env, agent, env_state, timestep, key
    )

    # Convert to numpy for indexing
    episode_length = int(episode_length)

    # Render the episode
    print(f"\nRendering episode (length: {episode_length})...")
    for i in range(episode_length):
        try:
            # Index PyTree properly
            state_i = jax.tree.map(lambda x: x[i], states)
            env.render(state_i)
        except NotImplementedError:
            if i == 0:
                print("Warning: render() not implemented for this environment")
            break

    # Final render
    try:
        env.render(final_env_state)
    except NotImplementedError:
        pass

    # Print episode statistics
    print(f"\nEpisode finished!")
    print(f"Episode length: {episode_length}")
    print(f"Total reward per agent: {episode_reward}")
    print(f"Mean reward: {jnp.mean(episode_reward):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play and render one episode using a trained agent")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint"
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Training step to load"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="Path to environment config yaml file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for episode (default: 0)"
    )

    args = parser.parse_args()

    play_episode(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        env_config_path=args.env_config,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
