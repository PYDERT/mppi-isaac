from dataclasses import dataclass, field
from mppiisaac.planner.mppi import MPPIConfig
from hydra.core.config_store import ConfigStore

from typing import List, Optional


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    goal: List[float]
    nx: int
    actors: List[str]
    initial_actor_positions: List[List[float]]


cs = ConfigStore.instance()
cs.store(name="config_point_robot", node=ExampleConfig)
cs.store(name="config_multi_point_robot", node=ExampleConfig)
cs.store(name="config_heijn_robot", node=ExampleConfig)
cs.store(name="config_boxer_robot", node=ExampleConfig)
cs.store(name="config_jackal_robot", node=ExampleConfig)
cs.store(name="config_multi_jackal", node=ExampleConfig)
cs.store(name="config_panda", node=ExampleConfig)
cs.store(name="config_omnipanda", node=ExampleConfig)
cs.store(name="config_panda_push", node=ExampleConfig)
cs.store(name="config_heijn_push", node=ExampleConfig)
cs.store(name="config_boxer_push", node=ExampleConfig)
cs.store(name="config_panda_c_space_goal", node=ExampleConfig)
cs.store(name="config_no_isaac", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
