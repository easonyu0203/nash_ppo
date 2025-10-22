"""
Unity Inference Server - Responds to decision requests from Unity with trained agent actions.

Usage:
    # Run with editor (default port 5004)
    JAX_PLATFORMS=cpu uv run scripts/unity_inference.py \
        --checkpoint-dir ./checkpoints/unity/3d_ball/ippo/run0 \
        --step 100 \
        --time-scale 1.0 \
        --seed 0

    # Run with build app
    JAX_PLATFORMS=cpu uv run scripts/unity_inference.py \
        --checkpoint-dir ./checkpoints/unity/3d_ball/ippo/run0 \
        --step 10000 \
        --file-name /path/to/build.app \
        --port 5005

    # Run with custom time scale
    JAX_PLATFORMS=cpu uv run scripts/unity_inference.py \
        --checkpoint-dir ./checkpoints/unity/3d_ball/ippo/run0 \
        --step 10000 \
        --time-scale 1.0
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
from typing import Optional
import jax
import numpy as np

from envs.wrappers.unity_env_wrapper import UnityEnvWrapper
from agents import BaseAgent


def run_inference_server(
    checkpoint_dir: str,
    step: int,
    file_name: Optional[str] = None,
    port: Optional[int] = None,
    time_scale: float = 20.0,
    seed: int = 0,
):
    """
    Run inference server that responds to Unity decision requests.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        step: Training step to load
        file_name: Path to Unity executable (None = Editor mode)
        port: Port for communication (None = auto-select)
        time_scale: Unity time scale
        seed: Random seed
    """
    # Load agent
    print(f"Loading checkpoint: {checkpoint_dir} at step {step}")
    key = jax.random.key(seed)
    agent = BaseAgent.load_checkpoint(checkpoint_dir, step, key)
    print(f"Loaded agent: {agent.get_class_name()}\n")

    # Connect to Unity
    print(f"Connecting to Unity...")
    print(f"  File: {file_name if file_name else 'Editor'}")
    print(f"  Port: {port if port else 'auto'}")
    print(f"  Time scale: {time_scale}")

    env = UnityEnvWrapper(
        file_name=file_name,
        num_areas=1,  # Single area for inference
        base_port=port,
        time_scale=time_scale,
        seed=seed,
        worker_id=0,
        no_graphics=True,
    )

    print(f"\nConnected! Server running...")
    print(f"  Agents: {env.num_agents}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print("\nWaiting for decision requests (Ctrl+C to stop)...\n")

    # Reset and start inference loop
    timestep = env.reset(seed=seed)
    step_count = 0

    # Track episode returns for each agent
    num_agents = env.num_agents
    episode_returns = np.zeros(num_agents)
    episode_lengths = np.zeros(num_agents, dtype=int)
    episode_count = 0

    try:
        while True:
            # Get action from agent for current observations
            key, action_key = jax.random.split(key)
            actions = agent.get_action(
                timestep.observation[0],  # Remove area dimension
                action_key,
                timestep.action_mask[0]
            )

            # Step environment (add area dimension back)
            actions_batched = np.array(actions)[np.newaxis, :]
            timestep = env.step(actions_batched)

            step_count += 1

            # Accumulate rewards and lengths
            episode_returns += timestep.reward[0]  # Remove area dimension
            episode_lengths += 1

            # Check for done agents (terminated or truncated)
            done = timestep.terminated[0] | timestep.truncated[0]
            if done.any():
                # Log returns for done agents
                for agent_idx in np.where(done)[0]:
                    episode_count += 1
                    print(f"Episode {episode_count} | Agent {agent_idx} | "
                          f"Return: {episode_returns[agent_idx]:.2f} | "
                          f"Length: {episode_lengths[agent_idx]}")

                    # Reset tracking for this agent
                    episode_returns[agent_idx] = 0
                    episode_lengths[agent_idx] = 0

    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    finally:
        env.close()
        print("Server closed.")


def main():
    parser = argparse.ArgumentParser(description="Unity inference server")

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
        "--file-name",
        type=str,
        default=None,
        help="Path to Unity executable (None = Editor mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for communication (None = auto-select)"
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=20.0,
        help="Unity time scale (default: 20.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )

    args = parser.parse_args()

    run_inference_server(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        file_name=args.file_name,
        port=args.port,
        time_scale=args.time_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
