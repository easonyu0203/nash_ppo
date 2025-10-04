"""Jumanji environment creators."""

from omegaconf import DictConfig
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from envs.wrappers import AutoResetWrapper, JumanjiWrapper


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

    # Wrap with JumanjiWrapper and AutoResetWrapper
    env = AutoResetWrapper(JumanjiWrapper(env))
    return env
