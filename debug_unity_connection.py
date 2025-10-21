"""
Debug script to diagnose Unity ML-Agents connection issues.

Usage:
    1. Open your Unity project and press Play
    2. Run this script: python debug_unity_connection.py
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import time
import logging
# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def print_separator(char="=", length=60):
    print(char * length)

def main():
    print_separator()
    print("Unity ML-Agents Connection Debugger")
    print_separator()

    # Initialize side channels
    engine_channel = EngineConfigurationChannel()

    engine_channel.set_configuration_parameters(
        time_scale=1.0,
        width=84,
        height=84,
        quality_level=0,
        target_frame_rate=-1,
    )


    print("\n[1] Attempting to connect to Unity Editor on port 5004...")
    print("    Make sure Unity is running and in Play mode!\n")

    try:
        unity_env = UnityEnvironment(
            file_name=None,  # None = Editor mode
            worker_id=0,
            base_port=5004,
            seed=42,
            no_graphics=False,
            side_channels=[engine_channel],
        )
        print("[✓] Successfully connected to Unity!")

    except Exception as e:
        print(f"[✗] Failed to connect to Unity!")
        print(f"    Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Unity Editor open?")
        print("  2. Is the scene in Play mode?")
        print("  3. Is port 5004 already in use?")
        print("  4. Try changing base_port to 5005 or another port")
        return

    # Check behavior specs
    print_separator()
    print("Checking Behaviors")
    print_separator()

    behavior_specs = unity_env.behavior_specs
    print(f"Number of behaviors found: {len(behavior_specs)}")

    if len(behavior_specs) == 0:
        print("\n[✗] No behaviors found!")
        print("\nThis means Unity ML-Agents can't find any Agent components.")
        print("\nPossible reasons:")
        print("  1. No GameObjects with 'Agent' component in the scene")
        print("  2. ML-Agents package not installed in Unity")
        print("  3. Agents not properly initialized")
        print("  4. Wrong scene loaded in Unity")

        print("\nTo fix:")
        print("  1. In Unity, check the Hierarchy for GameObjects with Agent components")
        print("  2. Make sure 'com.unity.ml-agents' package is installed")
        print("  3. Add a 'Behavior Name' in the Agent's Behavior Parameters component")
        print("  4. Check Unity Console for any errors")

        unity_env.close()
        return

    print("[✓] Found behaviors!\n")

    # Print detailed behavior information
    for behavior_name, spec in behavior_specs.items():
        print(f"Behavior: '{behavior_name}'")
        print("-" * 40)

        # Action spec
        print(f"Action Spec:")
        print(f"  Continuous actions: {spec.action_spec.continuous_size}")
        print(f"  Discrete branches:  {spec.action_spec.discrete_size}")
        if spec.action_spec.discrete_size > 0:
            print(f"  Branch sizes:       {spec.action_spec.discrete_branches}")

        # Observation specs
        print(f"Observation Specs:")
        for i, obs_spec in enumerate(spec.observation_specs):
            print(f"  Observation {i}: shape={obs_spec.shape}, type={obs_spec.observation_type}")

        print()

    # Try to get initial steps
    print_separator()
    print("Testing Environment Reset")
    print_separator()

    unity_env.reset()

    for behavior_name in behavior_specs.keys():
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)

        num_decision = len(decision_steps)
        num_terminal = len(terminal_steps)

        print(f"Behavior: '{behavior_name}'")
        print(f"  Decision agents: {num_decision}")
        print(f"  Terminal agents: {num_terminal}")

        if num_decision > 0:
            print(f"  Agent IDs: {decision_steps.agent_id}")

        print()

    # Close connection
    print_separator()
    print("Closing Connection")
    print_separator()
    unity_env.close()
    print("[✓] Connection closed successfully!")

    print_separator()
    print("Diagnostic Complete!")
    print_separator()

if __name__ == "__main__":
    main()
