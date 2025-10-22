"""
Unity Environment Inspector - Connect to Unity and inspect environment properties.

Usage:
    # Inspect Unity editor (default port 5004)
    python scripts/inspect_unity_env.py

    # Inspect with custom port
    python scripts/inspect_unity_env.py --port 5005

    # Inspect build app
    python scripts/inspect_unity_env.py --file-name /path/to/build.app --port 5005
"""

import argparse
from typing import Optional
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from envs.wrappers.unity_env_wrapper import UnityEnvWrapper


def inspect_unity_environment(
    file_name: Optional[str] = None,
    port: Optional[int] = None,
    worker_id: int = 0,
):
    """
    Connect to Unity and inspect environment properties.

    Args:
        file_name: Path to Unity executable (None = Editor mode)
        port: Port for communication (None = auto-select)
        worker_id: Worker ID for port offset
    """
    print("=" * 70)
    print("Unity Environment Inspector")
    print("=" * 70)

    # Connect to Unity
    print(f"\n[1] Connecting to Unity...")
    print(f"    File: {file_name if file_name else 'Editor Mode'}")
    print(f"    Port: {port if port else 'auto-select'}")

    env = UnityEnvironment(
        file_name=file_name,
        worker_id=worker_id,
        base_port=port,
        seed=0,
        no_graphics=True,
        num_areas=1,  # Single area for inspection
    )

    try:
        # Reset to get initial state
        env.reset()

        # Get behavior specs
        behavior_names = list(env.behavior_specs.keys())
        num_behaviors = len(behavior_names)

        print(f"\n[2] Behavior Information")
        print(f"    Number of behavior types: {num_behaviors}")
        print(f"    Behavior names: {behavior_names}")

        # Inspect each behavior
        total_agents = 0
        for i, behavior_name in enumerate(behavior_names):
            print(f"\n[3.{i+1}] Behavior: '{behavior_name}'")
            print("    " + "-" * 60)

            # Get behavior spec
            spec = env.behavior_specs[behavior_name]

            # Get current agents for this behavior
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            num_agents = len(decision_steps) + len(terminal_steps)
            total_agents += num_agents

            print(f"    Number of agents: {num_agents}")
            print(f"      - Decision steps: {len(decision_steps)}")
            print(f"      - Terminal steps: {len(terminal_steps)}")

            if num_agents > 0:
                print(f"    Agent IDs: {decision_steps.agent_id.tolist() + terminal_steps.agent_id.tolist()}")

            # Action space information
            print(f"\n    Action Space:")
            action_spec = spec.action_spec
            print(f"      - Continuous actions: {action_spec.continuous_size}")
            print(f"      - Discrete actions: {action_spec.discrete_size}")

            if action_spec.discrete_size > 0:
                branches = action_spec.discrete_branches
                print(f"      - Discrete branches: {branches}")

                # Convert to Gymnasium space
                if action_spec.discrete_size == 1:
                    print(f"      - Gymnasium equivalent: Discrete({int(branches[0])})")
                else:
                    branches_list = list(branches)
                    print(f"      - Gymnasium equivalent: MultiDiscrete({branches_list})")

            if action_spec.continuous_size > 0:
                print(f"      - Gymnasium equivalent: Box(shape=({action_spec.continuous_size},))")

            # Observation space information
            print(f"\n    Observation Space:")
            obs_specs = spec.observation_specs
            print(f"      - Number of observations: {len(obs_specs)}")

            for j, obs_spec in enumerate(obs_specs):
                print(f"      - Observation {j}:")
                print(f"          Shape: {obs_spec.shape}")
                print(f"          Dimension type: {obs_spec.observation_type}")
                print(f"          Gymnasium equivalent: Box(shape={obs_spec.shape}, dtype=float32)")

            # Sample observation if available
            if len(decision_steps) > 0:
                print(f"\n    Sample Observation (first agent):")
                sample_obs = decision_steps.obs[0][0]  # First observation type, first agent
                print(f"      - Shape: {sample_obs.shape}")
                print(f"      - Min/Max: [{sample_obs.min():.3f}, {sample_obs.max():.3f}]")
                print(f"      - Mean/Std: {sample_obs.mean():.3f} ± {sample_obs.std():.3f}")

                # Show first few values if small
                if sample_obs.size <= 20:
                    print(f"      - Values: {sample_obs.flatten()}")

        # Summary
        print(f"\n[4] Summary")
        print("    " + "-" * 60)
        print(f"    Total agents across all behaviors: {total_agents}")

        # Check if all behaviors have same spec
        if num_behaviors > 1:
            first_spec = env.behavior_specs[behavior_names[0]]
            all_same = True
            for name in behavior_names[1:]:
                spec = env.behavior_specs[name]
                if not specs_equal(first_spec, spec):
                    all_same = False
                    break

            if all_same:
                print(f"    All behaviors have identical action/observation spaces ✓")
            else:
                print(f"    WARNING: Behaviors have different action/observation spaces!")

        # Framework compatibility check
        print(f"\n[5] Framework Compatibility")
        print("    " + "-" * 60)

        issues = []
        for behavior_name in behavior_names:
            spec = env.behavior_specs[behavior_name]

            # Check continuous actions
            if spec.action_spec.continuous_size > 0:
                issues.append(f"Behavior '{behavior_name}' has continuous actions (not supported)")

            # Check multiple observations
            if len(spec.observation_specs) > 1:
                issues.append(f"Behavior '{behavior_name}' has {len(spec.observation_specs)} observations (only 1 supported)")

        if issues:
            print("    Issues found:")
            for issue in issues:
                print(f"      ✗ {issue}")
        else:
            print("    All checks passed ✓")
            print("    Environment is compatible with the framework!")

    finally:
        env.close()

    # Test with our UnityEnvWrapper
    print(f"\n[6] Testing with UnityEnvWrapper")
    print("    " + "-" * 60)
    print("    Creating wrapper to verify action/observation spaces...")

    try:
        wrapper = UnityEnvWrapper(
            file_name=file_name,
            num_areas=1,
            base_port=port,
            time_scale=1.0,
            seed=0,
            worker_id=worker_id,
            no_graphics=True,
        )

        print(f"\n    Wrapper created successfully ✓")
        print(f"      - num_agents: {wrapper.num_agents}")
        print(f"      - action_space: {wrapper.action_space}")
        print(f"      - observation_space: {wrapper.observation_space}")

        # Reset and check timestep structure
        timestep = wrapper.reset(seed=0)
        print(f"\n    TimeStep structure after reset:")
        print(f"      - observation.shape: {timestep.observation.shape}")
        print(f"      - action_mask.shape: {timestep.action_mask.shape}")
        print(f"      - reward.shape: {timestep.reward.shape}")
        print(f"      - terminated.shape: {timestep.terminated.shape}")
        print(f"      - truncated.shape: {timestep.truncated.shape}")

        print(f"\n    Sample observation (first agent):")
        sample_obs = timestep.observation[0, 0]  # First area, first agent
        print(f"      - Shape: {sample_obs.shape}")
        print(f"      - Min/Max: [{sample_obs.min():.3f}, {sample_obs.max():.3f}]")
        print(f"      - Mean/Std: {sample_obs.mean():.3f} ± {sample_obs.std():.3f}")
        if sample_obs.size <= 20:
            print(f"      - Values: {sample_obs}")

        print(f"\n    Sample action_mask (first agent):")
        sample_mask = timestep.action_mask[0, 0]  # First area, first agent
        print(f"      - Shape: {sample_mask.shape}")
        print(f"      - Values: {sample_mask}")

        wrapper.close()

    except Exception as e:
        print(f"\n    ✗ Failed to create wrapper: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("Inspection complete.")
    print("=" * 70)


def specs_equal(spec1, spec2) -> bool:
    """Check if two behavior specs are equal."""
    # Compare action specs
    if (spec1.action_spec.continuous_size != spec2.action_spec.continuous_size or
        spec1.action_spec.discrete_size != spec2.action_spec.discrete_size):
        return False

    if not np.array_equal(
        spec1.action_spec.discrete_branches,
        spec2.action_spec.discrete_branches
    ):
        return False

    # Compare observation specs
    if len(spec1.observation_specs) != len(spec2.observation_specs):
        return False

    for obs1, obs2 in zip(spec1.observation_specs, spec2.observation_specs):
        if obs1.shape != obs2.shape:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Unity environment properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
        help="Port for communication (None = auto-select: 5004 for editor, 5005 for builds)"
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID for port offset (default: 0)"
    )

    args = parser.parse_args()

    inspect_unity_environment(
        file_name=args.file_name,
        port=args.port,
        worker_id=args.worker_id,
    )


if __name__ == "__main__":
    main()
