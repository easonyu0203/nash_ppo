"""
Unity environment subprocess wrapper to avoid gRPC fork issues.

This wrapper creates the Unity environment in a separate subprocess,
avoiding the gRPC fork warning and thread issues that occur when
Unity environments are created in the main process.

Supports both synchronous (single instance) and asynchronous (multi-instance) modes.
"""

import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue
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
    num_areas: int,
    worker_id: int = 0,
    step_queue: Optional[Queue] = None
) -> None:
    """
    Worker process that runs the Unity environment.

    This function runs in a separate process and handles all Unity/gRPC
    interaction, avoiding fork issues in the main process.

    Args:
        conn: Pipe connection for communication with main process
        env_config: Environment configuration
        num_areas: Number of training areas
        worker_id: Worker ID for port offset and identification
        step_queue: Optional queue for async result reporting (multi-instance mode)
    """
    env = None
    try:
        # Import and create Unity environment HERE (in subprocess, after fork)
        from envs.wrappers.unity_env_wrapper import UnityEnvWrapper

        # Calculate unique seed for this worker
        base_seed = env_config.get("seed", 0)

        # Get base_port from config
        # If not specified, pass None to let UnityEnvironment auto-select:
        # - 5004 for editor (file_name=None)
        # - 5005 for builds (file_name specified)
        # UnityEnvironment will calculate actual port as: base_port + worker_id
        base_port = env_config.get("base_port", None)

        env = UnityEnvWrapper(
            file_name=env_config.get("file_name", None),
            num_areas=num_areas,
            base_port=base_port,  # None = auto-select, or explicit port from config
            time_scale=env_config.get("time_scale", 20.0),
            seed=base_seed + worker_id,  # Unique seed per worker
            worker_id=worker_id,          # Port offset: actual_port = base_port + worker_id
            no_graphics=env_config.get("no_graphics", True),
            additional_args=env_config.get("additional_args", None),
        )

        # Send success signal with environment info
        conn.send({
            "success": True,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "num_agents": env.num_agents,
            "worker_id": worker_id,
        })

        # Main loop: process commands from parent process
        while True:
            try:
                cmd, args = conn.recv()

                if cmd == UnitySubprocessCommand.RESET:
                    seed = args.get("seed") if args else None
                    timestep = env.reset(seed=seed)

                    # Send via queue if available (async mode), otherwise via conn (sync mode)
                    if step_queue is not None:
                        step_queue.put({
                            "success": True,
                            "worker_id": worker_id,
                            "timestep": timestep
                        })
                    else:
                        conn.send({"success": True, "timestep": timestep})

                elif cmd == UnitySubprocessCommand.STEP:
                    actions = args["actions"]
                    timestep = env.step(actions)

                    # Send via queue if available (async mode), otherwise via conn (sync mode)
                    if step_queue is not None:
                        step_queue.put({
                            "success": True,
                            "worker_id": worker_id,
                            "timestep": timestep
                        })
                    else:
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
                    response = {"success": False, "error": f"Unknown command: {cmd}"}
                    if step_queue is not None:
                        step_queue.put(response)
                    else:
                        conn.send(response)

            except Exception as e:
                response = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "worker_id": worker_id
                }
                if step_queue is not None:
                    step_queue.put(response)
                else:
                    conn.send(response)

    except Exception as e:
        # Failed to initialize environment
        conn.send({
            "success": False,
            "error": f"Failed to initialize Unity environment: {str(e)}",
            "traceback": traceback.format_exc(),
            "worker_id": worker_id
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

    Supports both sync mode (single instance) and async mode (multi-instance with queue).
    """

    def __init__(
        self,
        env_config: DictConfig,
        num_areas: int = 1,
        worker_id: int = 0,
        step_queue: Optional[Queue] = None,
        ctx: Optional[mp.context.BaseContext] = None,
        wait_for_init: bool = True
    ):
        """
        Initialize the subprocess wrapper.

        Args:
            env_config: Unity environment configuration
            num_areas: Number of training areas (parallel agents in Unity)
            worker_id: Worker ID for port offset and identification
            step_queue: Optional queue for async communication (multi-instance mode)
            ctx: Optional multiprocessing context (default: spawn)
            wait_for_init: If True, blocks until worker initializes. If False,
                          call wait_for_initialization() later to complete setup.
        """
        # Use 'spawn' instead of 'fork' to avoid any fork issues
        if ctx is None:
            ctx = mp.get_context('spawn')

        self._env_config = env_config
        self._num_areas = num_areas
        self._worker_id = worker_id
        self._step_queue = step_queue
        self._parent_conn, child_conn = ctx.Pipe()
        self._closed = False
        self._initialized = False

        # Start the subprocess
        self._process = ctx.Process(
            target=_unity_worker_process,
            args=(child_conn, env_config, num_areas, worker_id, step_queue),
            daemon=False  # Don't use daemon - we want clean shutdown
        )
        self._process.start()

        # Optionally wait for initialization
        if wait_for_init:
            self.wait_for_initialization()

    def wait_for_initialization(self) -> None:
        """
        Wait for the worker process to complete initialization.

        This method must be called if wait_for_init=False was passed to __init__.

        Raises:
            RuntimeError: If initialization failed or already initialized
        """
        if self._initialized:
            raise RuntimeError(f"Worker {self._worker_id} already initialized")

        # Wait for initialization response
        response = self._parent_conn.recv()
        if not response["success"]:
            self._process.terminate()
            self._process.join(timeout=5)
            raise RuntimeError(
                f"Failed to initialize Unity environment in subprocess (worker_id={self._worker_id}):\n"
                f"{response['error']}\n"
                f"{response.get('traceback', '')}"
            )

        # Store environment info
        self._observation_space = response["observation_space"]
        self._action_space = response["action_space"]
        self._num_agents = response["num_agents"]
        self._initialized = True

    def send_command(self, cmd: str, args: Any = None) -> None:
        """
        Send a command to the worker process (non-blocking).
        Used in async mode with queue - results will be in step_queue.

        Args:
            cmd: Command to send
            args: Command arguments
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call wait_for_initialization() first.")
        self._parent_conn.send((cmd, args))

    def reset(self, seed: Optional[int] = None, options: Optional[Any] = None) -> TimeStep:
        """
        Reset the environment.
        In sync mode (no queue), waits for response via conn.
        In async mode (with queue), should use send_command + poll queue instead.

        Args:
            seed: Random seed for environment
            options: Additional options (currently unused by Unity environments)
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call wait_for_initialization() first.")

        self._parent_conn.send((UnitySubprocessCommand.RESET, {"seed": seed}))

        # Sync mode: wait for response via conn
        if self._step_queue is None:
            response = self._parent_conn.recv()

            if not response["success"]:
                raise RuntimeError(
                    f"Environment reset failed:\n{response['error']}\n"
                    f"{response.get('traceback', '')}"
                )

            return response["timestep"]
        else:
            # Async mode: caller should poll queue
            # This method shouldn't be called directly in async mode
            raise RuntimeError(
                "reset() should not be called directly in async mode. "
                "Use send_command() and poll the step_queue instead."
            )

    def step(self, actions: Action) -> TimeStep:
        """
        Step the environment.
        In sync mode (no queue), waits for response via conn.
        In async mode (with queue), should use send_command + poll queue instead.
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call wait_for_initialization() first.")

        self._parent_conn.send((UnitySubprocessCommand.STEP, {"actions": actions}))

        # Sync mode: wait for response via conn
        if self._step_queue is None:
            response = self._parent_conn.recv()

            if not response["success"]:
                raise RuntimeError(
                    f"Environment step failed:\n{response['error']}\n"
                    f"{response.get('traceback', '')}"
                )

            return response["timestep"]
        else:
            # Async mode: caller should poll queue
            # This method shouldn't be called directly in async mode
            raise RuntimeError(
                "step() should not be called directly in async mode. "
                "Use send_command() and poll the step_queue instead."
            )

    @property
    def observation_space(self):
        """Get observation space."""
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call wait_for_initialization() first.")
        return self._observation_space

    @property
    def action_space(self):
        """Get action space."""
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call wait_for_initialization() first.")
        return self._action_space

    @property
    def num_agents(self) -> int:
        """Get number of agents."""
        if not self._initialized:
            raise RuntimeError("Worker not initialized. Call wait_for_initialization() first.")
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
        # Check if _closed exists (initialization may have failed)
        if hasattr(self, '_closed') and not self._closed:
            self.close()
