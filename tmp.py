from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from envs.wrappers import AutoResetWrapper, JumanjiWrapper

env = RobotWarehouse(generator = RandomGenerator(
column_height = 8,
  shelf_rows = 2,
  shelf_columns = 3,
  num_agents = 2,
  sensor_range = 1,
  request_queue_size = 2,
))

env = AutoResetWrapper(JumanjiWrapper(env))

print(env)