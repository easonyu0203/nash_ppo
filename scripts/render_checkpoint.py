"""
Render one episode using a trained agent checkpoint.

Usage:
    uv run scripts/render_checkpoint.py --checkpoint-dir ./checkpoints/gym/lunar_lander/ippo/default_run --step 2000 --env-config conf/env/gym/lunar_lander.yaml --seed 100 --fps 60
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
import time
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from envs import create_env
from agents import BaseAgent


def play_episode(checkpoint_dir: str, step: int, env_config_path: str, seed: int = 0, fps: float = 4.0):
    """
    Load a checkpoint and play one episode with rendering.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        step: Training step to load
        env_config_path: Path to environment config yaml
        seed: Random seed for episode
        fps: Frames per second for rendering (default: 4.0)
    """
    # Load agent from checkpoint
    print(f"Loading checkpoint from {checkpoint_dir} at step {step}...")
    key = jax.random.key(seed)
    agent = BaseAgent.load_checkpoint(checkpoint_dir, step, key)
    print(f"Loaded agent: {agent.get_class_name()}")

    # Load environment config
    env_config = OmegaConf.load(env_config_path)
    print(f"Environment config: {OmegaConf.to_yaml(env_config)}")

    # Create environment (without auto-reset for rendering)
    env = create_env(env_config, auto_reset=False, render_mode="human")
    print(f"Environment: {env}")
    print(f"Num agents: {env.num_agents}")

    # Reset environment
    timestep = env.reset(seed=seed)

    # Episode tracking
    print(f"\nRunning episode with rendering at {fps} FPS...")
    episode_reward = np.zeros(env.num_agents)
    step_count = 0
    max_steps = 1000  # Safety limit

    # FPS capping
    frame_delay = 1.0 / fps if fps > 0 else 0
    last_frame_time = time.time()

    # Try to render initial state
    try:
        env.render()
        if frame_delay > 0:
            time.sleep(frame_delay)
            last_frame_time = time.time()
    except NotImplementedError:
        print("Warning: render() not implemented for this environment")
        return

    # Run episode
    while not timestep.done.item() and step_count < max_steps:
        # Get action from agent
        key, action_key = jax.random.split(key)
        action_jax = agent.get_action(timestep.observation, action_key, timestep.action_mask)

        # Convert action to numpy for env.step()
        action_np = np.array(action_jax)

        # Step environment
        timestep = env.step(action_np)

        # Accumulate reward from this step
        episode_reward = episode_reward + np.array(timestep.reward)
        step_count += 1

        # Render current state
        env.render()

        # Cap FPS by sleeping if needed
        if frame_delay > 0:
            elapsed = time.time() - last_frame_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_frame_time = time.time()

    # Print episode statistics
    print(f"\nEpisode finished!")
    print(f"Episode length: {step_count}")
    # Format rewards with 2 decimal places
    reward_str = ", ".join([f"{float(r):.2f}" for r in episode_reward])
    print(f"Episode return per agent: [{reward_str}]")

    # Close environment
    env.close()

    # Wait for user to press Enter before closing
    input("\nPress Enter to close...")


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
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Frames per second for rendering (default: 4.0)"
    )

    args = parser.parse_args()

    play_episode(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        env_config_path=args.env_config,
        seed=args.seed,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
