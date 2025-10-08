"""Jumanji environment creators."""

from omegaconf import DictConfig
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from jumanji.environments.routing.connector.generator import RandomWalkGenerator
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfRandomGenerator
from jumanji.environments.routing.lbf.env import LevelBasedForaging
from envs.wrappers.jumanji_wrapper import RobotWarehouseWrapper, ConnectorWrapper, LbfWrapper


def create_robot_warehouse(config: DictConfig):
    """Create a Robot Warehouse environment from Jumanji.

    Args:
        config: Environment configuration with parameters:
            - column_height: Height of columns in the warehouse
            - shelf_rows: Number of shelf rows
            - shelf_columns: Number of shelf columns
            - num_agents: Number of robot agents
            - sensor_range: Sensor range for agents
            - request_queue_size: Size of request queue

    Returns:
        Wrapped RobotWarehouse environment
    """
    generator = RandomGenerator(
        column_height=config.column_height,
        shelf_rows=config.shelf_rows,
        shelf_columns=config.shelf_columns,
        num_agents=config.num_agents,
        sensor_range=config.sensor_range,
        request_queue_size=config.request_queue_size,
    )
    env = RobotWarehouse(generator=generator)

    # Wrap with RobotWarehouseWrapper
    env = RobotWarehouseWrapper(env)
    return env


def create_connector(config: DictConfig):
    """Create a Connector environment from Jumanji.

    Args:
        config: Environment configuration with parameters:
            - grid_size: Size of the square grid
            - num_agents: Number of agents/paths

    Returns:
        Wrapped Connector environment
    """
    generator = RandomWalkGenerator(
        grid_size=config.grid_size,
        num_agents=config.num_agents,
    )
    env = Connector(generator=generator)

    # Wrap with ConnectorWrapper
    env = ConnectorWrapper(env)
    return env


def create_lbf(config: DictConfig):
    """Create a Level Based Foraging (LBF) environment from Jumanji.

    Args:
        config: Environment configuration with parameters:
            - grid_size: Size of the grid (must be >= 5)
            - num_agents: Number of agents (must be > 0)
            - num_food: Number of food items (must be > 0)
            - fov: Field of view (between 1 and grid_size)
            - max_agent_level: Maximum agent level (default: 2)
            - force_coop: Whether to force cooperation (default: False)
            - time_limit: Episode time limit (default: 100)
            - normalize_reward: Whether to normalize rewards (default: True)
            - penalty: Penalty for invalid actions (default: 0.0)

    Returns:
        Wrapped LBF environment with GridObserver
    """
    generator = LbfRandomGenerator(
        grid_size=config.grid_size,
        num_agents=config.num_agents,
        num_food=config.num_food,
        fov=config.fov,
        max_agent_level=config.get("max_agent_level", 2),
        force_coop=config.get("force_coop", False),
    )
    env = LevelBasedForaging(
        generator=generator,
        grid_observation=True,  # Always use GridObserver
        time_limit=config.get("time_limit", 100),
        normalize_reward=config.get("normalize_reward", True),
        penalty=config.get("penalty", 0.0),
    )

    # Wrap with LbfWrapper
    env = LbfWrapper(env)
    return env
