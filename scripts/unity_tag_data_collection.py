"""
Unity Tag Data Collection - Collect episode data for statistical analysis.

Usage:
    uv run scripts/unity_tag_data_collection.py \
        --checkpoint-dir ./checkpoints/unity/tag/run0 \
        --step-start 0 \
        --step-end 20000 \
        --step-interval 1000 \
        --target-timesteps 1000 \
        --num-runs 100 \
        --output-name run0 \
        --file-name /Users/Ethan/Developer/Projects/Usyd/research/external/ml-agents/Project/Builds/inference/main.app \
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.wrappers.unity_env_wrapper import UnityEnvWrapper
from agents import BaseAgent, StatefulAgent



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
    target_timesteps: int,
    pbar: Optional[tqdm] = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Continuously run and collect multiple episodes.

    Logic:
    - When in done state, don't record action/reward
    - When episode ends (act and it becomes done), record that timestep
    - Each episode must have >= target_timesteps data points

    Args:
        pbar: Progress bar for episode collection (will be updated). Optional for parallel execution.

    Returns:
        List of (actions_df, rewards_df) tuples, truncated to target_timesteps
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
                # Truncate to target_timesteps
                actions_array = actions_list[:target_timesteps]
                rewards_array = rewards_list[:target_timesteps]

                # Assert we have enough data
                assert len(actions_array) >= target_timesteps, \
                    f"Episode {episode_count} only has {len(actions_array)} timesteps, expected >= {target_timesteps}"

                actions_df = pd.DataFrame(actions_array, columns=agent_names)
                rewards_df = pd.DataFrame(rewards_array, columns=agent_names)
                all_episodes.append((actions_df, rewards_df))

                # Reset for next episode
                actions_list = []
                rewards_list = []
                episode_count += 1
                if pbar is not None:
                    pbar.update(1)

                # Reset carry for next episode
                carry = agent.initialize_carry(num_agents)

        prev_all_done = all_done

    return all_episodes


def process_checkpoint(
    checkpoint_dir: str,
    step: int,
    checkpoint_idx: int,
    num_runs: int,
    seed: int,
    target_timesteps: int,
    file_name: Optional[str],
    base_port: Optional[int],
    time_scale: float,
    output_name: str,
) -> tuple[int, bool, str]:
    """
    Process a single checkpoint: load agent, collect episodes, save data.

    Args:
        output_name: Name of the output subdirectory within data/

    Returns:
        Tuple of (step, success, message)
    """
    try:
        # Load agent
        key = jax.random.key(seed)
        agent = BaseAgent.load_checkpoint(checkpoint_dir, step, key)

        # Create output directory
        output_dir = Path("data") / output_name / f"step_{step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate port for this checkpoint
        current_port = base_port + checkpoint_idx if base_port is not None else None

        # Create Unity environment
        env = UnityEnvWrapper(
            file_name=file_name,
            num_areas=1,
            base_port=current_port,
            time_scale=time_scale,
            seed=seed,
            worker_id=checkpoint_idx,
            no_graphics=True,
        )

        try:
            # Collect all episodes
            key = jax.random.key(seed)
            all_episodes = collect_all_episodes(
                env, agent, key, num_runs, seed, target_timesteps, pbar=None
            )

            # Save each episode to CSV
            for run_id, (actions_df, rewards_df) in enumerate(all_episodes):
                run_dir = output_dir / f"run_{run_id}"
                run_dir.mkdir(exist_ok=True)

                # Save raw data without headers or index
                actions_df.to_csv(run_dir / "actions.csv", index=False, header=False)
                rewards_df.to_csv(run_dir / "rewards.csv", index=False, header=False)

            return (step, True, f"Completed {num_runs} episodes")

        finally:
            env.close()

    except Exception as e:
        return (step, False, f"Error: {str(e)}")


def run_data_collection(
    checkpoint_dir: str,
    step_start: int,
    step_end: int,
    step_interval: int,
    num_runs: int,
    seed: int,
    target_timesteps: int,
    output_name: str,
    file_name: Optional[str] = None,
    port: Optional[int] = None,
    time_scale: float = 20.0,
    max_workers: int = 4,
):
    """
    Collect episode data for multiple checkpoints and runs in parallel.

    Args:
        checkpoint_dir: Directory containing the checkpoints
        step_start: First training step to collect data from
        step_end: Last training step to collect data from (inclusive)
        step_interval: Interval between checkpoint steps
        num_runs: Number of episodes to collect per checkpoint
        seed: Base random seed
        target_timesteps: Target number of timesteps to collect per episode
        output_name: Name of the output subdirectory within data/
        file_name: Path to Unity executable (None = Editor mode)
        port: Port for communication (None = auto-select)
        time_scale: Unity time scale
        max_workers: Maximum number of parallel workers
    """
    # Generate checkpoint steps
    checkpoint_steps = list(range(step_start, step_end + 1, step_interval))

    print(f"Data Collection Configuration:")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Steps: {step_start} to {step_end} (interval: {step_interval})")
    print(f"  Total checkpoints: {len(checkpoint_steps)}")
    print(f"  Episodes per checkpoint: {num_runs}")
    print(f"  Total episodes: {len(checkpoint_steps) * num_runs}")
    print(f"  Timesteps per episode: {target_timesteps}")
    print(f"  Output directory: data/{output_name}/")
    print(f"  Base seed: {seed}")
    print(f"  Unity file: {file_name if file_name else 'Editor'}")
    print(f"  Port: {port if port else 'auto'}")
    print(f"  Time scale: {time_scale}")
    print(f"  Parallel workers: {max_workers}\n")

    try:
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all checkpoint tasks
            futures = {}
            for checkpoint_idx, step in enumerate(checkpoint_steps):
                future = executor.submit(
                    process_checkpoint,
                    checkpoint_dir,
                    step,
                    checkpoint_idx,
                    num_runs,
                    seed,
                    target_timesteps,
                    file_name,
                    port,
                    time_scale,
                    output_name,
                )
                futures[future] = step

            # Track progress as tasks complete
            with tqdm(total=len(checkpoint_steps), desc="Checkpoints") as pbar:
                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        result_step, success, message = future.result()
                        if success:
                            pbar.set_postfix_str(f"Step {result_step}: {message}")
                        else:
                            pbar.set_postfix_str(f"Step {result_step}: FAILED - {message}")
                    except Exception as e:
                        pbar.set_postfix_str(f"Step {step}: Exception - {str(e)}")
                    pbar.update(1)

        print(f"\n✓ Data collection complete!")
        print(f"  Collected {len(checkpoint_steps)} checkpoints × {num_runs} episodes = {len(checkpoint_steps) * num_runs} total episodes")
        print(f"  Data saved to: data/{output_name}/")

    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user...")
        print("Cleaning up...")


def main():
    parser = argparse.ArgumentParser(description="Unity Tag data collection")

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing the checkpoints"
    )
    parser.add_argument(
        "--step-start",
        type=int,
        required=True,
        help="First training step to collect data from"
    )
    parser.add_argument(
        "--step-end",
        type=int,
        required=True,
        help="Last training step to collect data from (inclusive)"
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        required=True,
        help="Interval between checkpoint steps"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        required=True,
        help="Number of episodes to collect per checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Base random seed"
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
        "--target-timesteps",
        type=int,
        default=1000,
        help="Target number of timesteps to collect per episode (default: 1000)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name of the output subdirectory within data/ (e.g., 'run0', 'experiment1')"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    run_data_collection(
        checkpoint_dir=args.checkpoint_dir,
        step_start=args.step_start,
        step_end=args.step_end,
        step_interval=args.step_interval,
        num_runs=args.num_runs,
        seed=args.seed,
        target_timesteps=args.target_timesteps,
        output_name=args.output_name,
        file_name=args.file_name,
        port=args.port,
        time_scale=args.time_scale,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
