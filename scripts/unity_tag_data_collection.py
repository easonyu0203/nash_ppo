"""
Unity Tag Data Collection - Collect episode data for statistical analysis.

Usage:
    uv run scripts/unity_tag_data_collection.py \
        --checkpoint-dir ./checkpoints/unity/tag/run0 \
        --step 10000 \
        --num-runs 100 \
        --seed 42 \
        --time-scale 20.0
"""

import os
import logging
import warnings
os.environ["XLA_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('orbax').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Sharding info not provided.*")

import argparse
from pathlib import Path
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.wrappers.unity_env_wrapper import UnityEnvWrapper
from agents import BaseAgent, StatefulAgent

# Maximum number of timesteps to collect per episode
MAX_TIMESTEPS = 1000


def action_to_scalar(action: np.ndarray) -> int:
    """Convert (2,) action to scalar using 3*action[0] + action[1]."""
    if action.shape == (2,):
        # Each element is 0-2, encode as scalar: 3*x + y
        return int(3 * action[0] + action[1])
    else:
        raise ValueError(f"Expected action shape (2,), got {action.shape}")


def collect_all_episodes(
    env: UnityEnvWrapper,
    agent: StatefulAgent,
    key: jax.random.PRNGKey,
    num_episodes: int,
    initial_seed: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Continuously run and collect multiple episodes.

    Logic:
    - When in done state, don't record action/reward
    - When episode ends (act and it becomes done), record that timestep
    - Each episode must have >= MAX_TIMESTEPS data points

    Returns:
        List of (actions_df, rewards_df) tuples, truncated to MAX_TIMESTEPS
    """
    num_agents = env.num_agents

    # Agent names: prey_0 to prey_7, predator_0, predator_1
    agent_names = [f"prey_{i}" for i in range(8)] + [f"predator_{i}" for i in range(2)]

    # Initial reset
    timestep = env.reset(seed=initial_seed)
    carry = agent.initialize_carry(num_agents)

    # Storage
    all_episodes = []
    actions_list = []
    rewards_list = []
    episode_count = 0

    # Track previous done state
    prev_all_done = False

    with tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        while episode_count < num_episodes:
            # Get action from agent
            key, action_key = jax.random.split(key)
            obs = jax.tree.map(lambda x: x[0], timestep.observation)
            obs_jax = jax.tree.map(jnp.asarray, obs)

            actions, carry = agent.get_action(obs_jax, carry, action_key)

            # Convert actions to scalars
            actions_np = np.array(actions)
            action_scalars = [action_to_scalar(actions_np[i]) for i in range(num_agents)]

            # Step environment
            actions_batched = actions_np[np.newaxis, :]
            timestep = env.step(actions_batched)

            # Get rewards
            rewards = timestep.reward[0]

            # Check if all agents are done
            terminated = timestep.terminated[0]
            truncated = timestep.truncated[0]
            all_done = np.all(terminated | truncated)

            # Record logic:
            # - If we were NOT in done state before, record this timestep
            # - If episode just ended (all_done=True), this is the last timestep to record
            if not prev_all_done:
                actions_list.append(action_scalars)
                rewards_list.append(rewards)

                # If episode ended, save it
                if all_done:
                    # Truncate to MAX_TIMESTEPS
                    actions_array = actions_list[:MAX_TIMESTEPS]
                    rewards_array = rewards_list[:MAX_TIMESTEPS]

                    # Assert we have enough data
                    assert len(actions_array) >= MAX_TIMESTEPS, \
                        f"Episode {episode_count} only has {len(actions_array)} timesteps, expected >= {MAX_TIMESTEPS}"

                    actions_df = pd.DataFrame(actions_array, columns=agent_names)
                    rewards_df = pd.DataFrame(rewards_array, columns=agent_names)
                    all_episodes.append((actions_df, rewards_df))

                    # Reset for next episode
                    actions_list = []
                    rewards_list = []
                    episode_count += 1
                    pbar.update(1)

                    # Reset carry for next episode
                    carry = agent.initialize_carry(num_agents)

            prev_all_done = all_done

    return all_episodes


def run_data_collection(
    checkpoint_dir: str,
    step: int,
    num_runs: int,
    seed: int,
    file_name: Optional[str] = None,
    port: Optional[int] = None,
    time_scale: float = 20.0,
):
    """
    Collect episode data for multiple runs.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        step: Training step to load
        num_runs: Number of episodes to collect
        seed: Base random seed (each run uses seed + run_id)
        file_name: Path to Unity executable (None = Editor mode)
        port: Port for communication (None = auto-select)
        time_scale: Unity time scale
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
        num_areas=1,
        base_port=port,
        time_scale=time_scale,
        seed=seed,
        worker_id=0,
        no_graphics=True,
    )

    print(f"\nConnected! Starting data collection...")
    print(f"  Agents: {env.num_agents}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Agent type: Stateful (assumed)")

    # Create output directory
    output_dir = Path("data") / f"step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Collect all episodes in one continuous run
    try:
        key = jax.random.key(seed)
        all_episodes = collect_all_episodes(env, agent, key, num_runs, seed)

        # Save each episode to CSV
        print(f"\nSaving episodes to disk...")
        for run_id, (actions_df, rewards_df) in enumerate(all_episodes):
            run_dir = output_dir / f"run_{run_id}"
            run_dir.mkdir(exist_ok=True)

            actions_df.to_csv(run_dir / "actions.csv", index_label="timestep")
            rewards_df.to_csv(run_dir / "rewards.csv", index_label="timestep")

            # Print summary for first few runs
            if run_id < 3:
                print(f"  Run {run_id}:")
                print(f"    Episode length: {len(actions_df)} timesteps")
                print(f"    Total rewards: predator_0={rewards_df['predator_0'].sum():.0f}, "
                      f"predator_1={rewards_df['predator_1'].sum():.0f}, "
                      f"prey_0={rewards_df['prey_0'].sum():.0f}")

        print(f"\n✓ Data collection complete!")
        print(f"  Collected {len(all_episodes)} episodes")
        print(f"  Data saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user...")
    finally:
        env.close()
        print("Environment closed.")


def main():
    parser = argparse.ArgumentParser(description="Unity Tag data collection")

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
        "--num-runs",
        type=int,
        required=True,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Base random seed (each run uses seed + run_id)"
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

    args = parser.parse_args()

    run_data_collection(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        num_runs=args.num_runs,
        seed=args.seed,
        file_name=args.file_name,
        port=args.port,
        time_scale=args.time_scale,
    )


if __name__ == "__main__":
    main()
