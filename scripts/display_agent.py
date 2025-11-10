"""
Display agent architecture from config file using nnx.display.

Usage:
    uv run scripts/display_agent.py conf/agent/gym/lstm_acrobot.yaml
    uv run scripts/display_agent.py conf/agent/gym/acrobot.yaml
    uv run scripts/display_agent.py conf/agent/unity/soccer_mlp.yaml
"""

import argparse
from pathlib import Path

import jax
from flax import nnx
from omegaconf import OmegaConf


from agents import create_agent


def display_agent_architecture(config_path: str, batch_size: int = 4):
    """Load agent config and display architecture.

    Args:
        config_path: Path to agent config yaml file
        batch_size: Batch size for carry initialization (stateful agents only)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    config = OmegaConf.load(config_path)

    # Create agent
    key = jax.random.key(0)
    agent = create_agent(config, key)

    # Display architecture
    print(f"\n{'='*80}")
    print(f"Agent Architecture: {config_path.name}")
    print(f"{'='*80}\n")
    print(nnx.display(agent))

    # Display carry info for stateful agents
    from agents import StatefulAgent
    if isinstance(agent, StatefulAgent):
        print(f"\n{'='*80}")
        print(f"Hidden State (Carry) Structure")
        print(f"{'='*80}\n")
        carry = agent.initialize_carry(batch_size)

        def print_carry_shape(path, value):
            path_str = '.'.join(str(k.key) if hasattr(k, 'key') else str(k) for k in path)
            shape_str = f"{value.shape} {value.dtype}"
            print(f"  {path_str:40s} {shape_str}")

        print(f"Batch size: {batch_size}\n")
        jax.tree_util.tree_map_with_path(print_carry_shape, carry)


def main():
    parser = argparse.ArgumentParser(
        description="Display agent architecture from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/display_agent.py conf/agent/gym/lstm_acrobot.yaml
  uv run scripts/display_agent.py conf/agent/gym/acrobot.yaml --batch-size 8
        """
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to agent config yaml file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for carry initialization (default: 4)"
    )

    args = parser.parse_args()
    display_agent_architecture(args.config, args.batch_size)


if __name__ == "__main__":
    main()