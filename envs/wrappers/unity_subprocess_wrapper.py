"""
Unity environment subprocess wrapper to avoid gRPC fork issues.

This wrapper creates the Unity environment in a separate subprocess,
avoiding the gRPC fork warning and thread issues that occur when
Unity environments are created in the main process.
"""

import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Optional, Tuple, Any
import traceback
import logging

import numpy as np
from omegaconf import DictConfig

from envs.mytypes import BaseEnv, TimeStep, Action


logger = logging.getLogger(__name__)


class UnitySubprocessCommand:
    """Commands for subprocess communication."""
    RESET = "reset"
    STEP = "step"
    CLOSE = "close"
    GET_SPACES = "get_spaces"
    GET_NUM_AGENTS = "get_num_agents"


def _unity_worker_process(
    conn: Connection,
    env_config: DictConfig,
    num_areas: int
) -> None:
    """
    Worker process that runs the Unity environment.

    This function runs in a separate process and handles all Unity/gRPC
    interaction, avoiding fork issues in the main process.

    Args:
        conn: Pipe connection for communication with main process
        env_config: Environment configuration
        num_areas: Number of training areas
    """
    env = None
    try:
        # Import and create Unity environment HERE (in subprocess, after fork)
        from envs.wrappers.unity_env_wrapper import UnityEnvWrapper

        env = UnityEnvWrapper(
            file_name=env_config.get("file_name", None),
            num_areas=num_areas,
            base_port=env_config.get("base_port", 5004),
            time_scale=env_config.get("time_scale", 20.0),
            seed=env_config.get("seed", 0),
            worker_id=env_config.get("worker_id", 0),
            no_graphics=env_config.get("no_graphics", True),
            additional_args=env_config.get("additional_args", None),
        )

        # Send success signal with environment info
        conn.send({
            "success": True,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "num_agents": env.num_agents,
        })

        # Main loop: process commands from parent process
        while True:
            try:
                cmd, args = conn.recv()

                if cmd == UnitySubprocessCommand.RESET:
                    seed = args.get("seed") if args else None
                    timestep = env.reset(seed=seed)
                    conn.send({"success": True, "timestep": timestep})

                elif cmd == UnitySubprocessCommand.STEP:
                    actions = args["actions"]
                    timestep = env.step(actions)
                    conn.send({"success": True, "timestep": timestep})

                elif cmd == UnitySubprocessCommand.GET_SPACES:
                    conn.send({
                        "success": True,
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                    })

                elif cmd == UnitySubprocessCommand.GET_NUM_AGENTS:
                    conn.send({"success": True, "num_agents": env.num_agents})

                elif cmd == UnitySubprocessCommand.CLOSE:
                    break

                else:
                    conn.send({"success": False, "error": f"Unknown command: {cmd}"})

            except Exception as e:
                conn.send({
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

    except Exception as e:
        # Failed to initialize environment
        conn.send({
            "success": False,
            "error": f"Failed to initialize Unity environment: {str(e)}",
            "traceback": traceback.format_exc()
        })
    finally:
        # Cleanup
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing Unity environment in subprocess: {e}")
        conn.close()


class UnitySubprocessWrapper(BaseEnv):
    """
    Wrapper that runs Unity environment in a subprocess.

    This avoids the gRPC fork warning by ensuring Unity/gRPC
    is only initialized in the subprocess, never in the main process.
    """

    def __init__(self, env_config: DictConfig, num_areas: int = 1):
        """
        Initialize the subprocess wrapper.

        Args:
            env_config: Unity environment configuration
            num_areas: Number of training areas (parallel agents in Unity)
        """
        # Use 'spawn' instead of 'fork' to avoid any fork issues
        ctx = mp.get_context('spawn')

        self._env_config = env_config
        self._num_areas = num_areas
        self._parent_conn, child_conn = ctx.Pipe()

        # Start the subprocess
        self._process = ctx.Process(
            target=_unity_worker_process,
            args=(child_conn, env_config, num_areas),
            daemon=False  # Don't use daemon - we want clean shutdown
        )
        self._process.start()

        # Wait for initialization response
        response = self._parent_conn.recv()
        if not response["success"]:
            self._process.terminate()
            self._process.join(timeout=5)
            raise RuntimeError(
                f"Failed to initialize Unity environment in subprocess:\n"
                f"{response['error']}\n"
                f"{response.get('traceback', '')}"
            )

        # Store environment info
        self._observation_space = response["observation_space"]
        self._action_space = response["action_space"]
        self._num_agents = response["num_agents"]
        self._closed = False

    def reset(self, seed: Optional[int] = None) -> TimeStep:
        """Reset the environment."""
        if self._closed:
            raise RuntimeError("Environment is closed")

        self._parent_conn.send((UnitySubprocessCommand.RESET, {"seed": seed}))
        response = self._parent_conn.recv()

        if not response["success"]:
            raise RuntimeError(
                f"Environment reset failed:\n{response['error']}\n"
                f"{response.get('traceback', '')}"
            )

        return response["timestep"]

    def step(self, actions: Action) -> TimeStep:
        """Step the environment."""
        if self._closed:
            raise RuntimeError("Environment is closed")

        self._parent_conn.send((UnitySubprocessCommand.STEP, {"actions": actions}))
        response = self._parent_conn.recv()

        if not response["success"]:
            raise RuntimeError(
                f"Environment step failed:\n{response['error']}\n"
                f"{response.get('traceback', '')}"
            )

        return response["timestep"]

    @property
    def observation_space(self):
        """Get observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """Get action space."""
        return self._action_space

    @property
    def num_agents(self) -> int:
        """Get number of agents."""
        return self._num_agents

    def render(self):
        """Render not supported in subprocess mode."""
        logger.warning("Render not supported in Unity subprocess wrapper")
        return None

    def close(self) -> None:
        """Close the environment and subprocess."""
        if self._closed:
            return

        self._closed = True

        try:
            # Send close command
            self._parent_conn.send((UnitySubprocessCommand.CLOSE, None))
            # Wait for process to finish
            self._process.join(timeout=10)

            # Force terminate if still alive
            if self._process.is_alive():
                logger.warning("Unity subprocess did not terminate cleanly, forcing termination")
                self._process.terminate()
                self._process.join(timeout=5)

                # Last resort: kill
                if self._process.is_alive():
                    logger.error("Unity subprocess still alive after terminate, killing")
                    self._process.kill()
                    self._process.join()

        except Exception as e:
            logger.error(f"Error closing Unity subprocess: {e}")
            try:
                self._process.kill()
            except:
                pass
        finally:
            try:
                self._parent_conn.close()
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if not self._closed:
            self.close()
