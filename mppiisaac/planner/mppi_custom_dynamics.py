from mppiisaac.planner.mppi import MPPIPlanner
from typing import Callable

import torch

torch.set_printoptions(precision=2, sci_mode=False)


class MPPICustomDynamicsPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, dynamics: Callable, obstacles):
        self.cfg = cfg
        self.objective = objective
        self.dynamics = dynamics
        self.obstacles = obstacles

        self.mppi = MPPIPlanner(
            cfg,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            obstacles=self.obstacles
        )
        self.current_state = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))
    
    def update_objective(self, objective):
        self.objective = objective

    def running_cost(self, state, t=None):
        return self.objective.compute_cost(state, t=t)

    def compute_action(self, state):
        self.current_state = torch.tensor([state], dtype=torch.float32, device=self.cfg.mppi.device)
        actions = self.mppi.command(self.current_state).cpu()
        return actions
