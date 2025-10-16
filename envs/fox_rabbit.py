"""
FoxRabbit Environment
=====================

A multi-agent predator-prey gridworld environment featuring *Fox* (predator) and *Rabbit* (prey) agents.
The environment is designed for MARL research with asymmetric observation ranges, multi-discrete actions,
local perception, and environmental interaction (scents and blocks).

Overview
--------
- **World:** Square grid where each cell may contain blocks, scents, or agents.
- **Agents:** Foxes and Rabbits.
- **Goal:** Foxes aim to capture Rabbits; Rabbits aim to survive as long as possible.
- **Episode Termination:** When all Rabbits are captured or the time limit is reached.

Core Rules
-----------
- Fox captures a Rabbit if they occupy the same cell.
- Captured Rabbits are removed from the grid and can only take "no scent" and "no move" actions.
- Both species can move, leave scents, place blocks, and mine (destroy) opposite-type blocks.
- Agents can pass through their own blocks, but must spend one move to leave a block cell.
- Scents decay over time based on a decay factor.

Action Space
-------------
Each agent has a **multi-discrete action space** consisting of two components:

1. **Scent Action (Discrete[4])**
   - 0: No scent
   - 1–3: Different scent types (configurable)
   - Leaves a scent in the current cell that decays each step.

2. **Interaction Action (Discrete[7])**
   - 0: Move Up
   - 1: Move Down
   - 2: Move Left
   - 3: Move Right
   - 4: No move
   - 5: Mine (destroy opposite-type block if present)
   - 6: Place block (of own type)

**Action Mask:** Shape (4, 7).  
Specifies which scent–interaction combinations are currently valid (e.g., mining only valid if a block exists).

Observation Space
------------------
Each agent observes a square region centered on itself, limited by its field of view (FOV).

**Shape:** (2 * fov + 1, 2 * fov + 1, C)

**Channels (C):**
1–4. Fox scent maps  
5–8. Rabbit scent maps  
9. Fox positions (binary)  
10. Rabbit positions (binary)  
11. Fox block map  
12. Rabbit block map  
13. Fog of world mask (1 = can see, 0 = fog)
14. Agent type indicator (1 for Fox, 0 for Rabbit)  
15. Accessibility mask (1 = valid cell, 0 = out of bounds)

The number of scent channels may vary with configuration.

Rewards
--------
| Event | Fox Reward | Rabbit Reward |
|--------|-------------|----------------|
| Rabbit captured | +1 | -1 |
| Rabbit alive per step | 0 | +0.001 |
| Default step | 0 | 0 |

The total reward for an agent is the sum of these event-based components.

Configuration Parameters
-------------------------
| Parameter | Description | Default |
|------------|--------------|----------|
| `grid_size` | Grid dimension (N × N) | 16 |
| `fov` | Agent field of view radius | 2 |
| `num_scent_types` | Number of scent types (including "no scent") | 4 |
| `allow_blocks` | Whether block placement/mining is enabled | True |
| `scent_decay_factor` | Scent decay multiplier per step | 0.9 |
| `rabbit_alive_reward` | Reward per alive step for Rabbits | 0.001 |
| `time_limit` | Maximum number of steps per episode | 1000 |
| `num_fox` | Number of fox in env | 2 |
| `num_rabbit` | Number of rabbit in env | 6 |

Episode Termination
-------------------
- When all Rabbits are captured, **or**
- When the time limit is reached.

"""

"""

"""

from functools import cached_property
from envs.myspaces import Space, MultiDiscrete, Box
from envs.mytypes import BaseEnv


class FoxRabbit(BaseEnv):

    def __init__(
        self, grid_size = 16, fox_fov = 2, rabbit_fov = 4, num_scent_types = 4,
        allow_blocks = True, scent_decay_factor = 0.9, rabbit_alive_reward = 0.001,
        time_limit = 1000, num_fox = 2, num_rabbit = 6,
    ):
        self.grid_size = grid_size
        self.fox_fov = fox_fov
        self.rabbit_fov = rabbit_fov
        self.num_scent_types = num_scent_types
        self.allow_blocks = allow_blocks
        self.scent_decay_factor = scent_decay_factor
        self.rabbit_alive_reward = rabbit_alive_reward
        self.time_limit = time_limit
        self.num_fox = num_fox
        self.num_rabbit = num_rabbit
        self.max_scent_strength = max(self.num_fox, self.num_rabbit)
        self.max_fov = max(fox_fov, rabbit_fov)
        self.max_fov_size = 2 * self.max_fov + 1

    def __repr__(self) -> str:
        return "Fox-Rabbit"
    
    @cached_property
    def action_space(self) -> Space:
        return MultiDiscrete([self.num_scent_types, 7])

    @cached_property
    def observation_space(self) -> Space:
        return Box(low=0.0, high=1.0, shape=(self.max_fov_size, self.max_fov_size, 14))