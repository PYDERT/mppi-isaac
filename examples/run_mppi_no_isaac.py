from mppiisaac.planner.mppi_no_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import torch
import zerorpc
import pytorch3d.transforms
import numpy as np

from mppiisaac.utils.config_store_no_isaac import ExampleConfig
from mppiisaac.planner.obstacle_class import Obstacle

import matplotlib.pyplot as plt

from typing import List

class Objective(object):
    def __init__(self, cfg, obstacles: List[Obstacle]):
        
        self.cfg = cfg
        self.goal_euler = torch.tensor(cfg.goal[:2], device=cfg.mppi.device)
        self.goal_orientation = torch.tensor([cfg.goal[2]], device=cfg.mppi.device)
        
        self.obstacles = obstacles

        self.success = False
        self.count = 0

        self.control = torch.zeros((len(cfg.mppi.noise_sigma)), device=cfg.mppi.device)
        self.current_state = torch.zeros((cfg.nx), device=cfg.mppi.device)

    def compute_cost(self, states, control):
        
        # Normlisation value
        # norm = torch.linalg.norm(self.goal_euler - self.current_state[:2]) * 
        
        # Distances robot
        total_cost = torch.linalg.norm(self.goal_euler - states[:, :2], dim=1, keepdim=True).squeeze(1)
        # total_cost_large = total_cost[total_cost > 1]**2
        # total_cost_small = total_cost[total_cost <= 1]**0.5
        # total_cost = torch.zeros_like(total_cost) + total_cost[total_cost > 1]**2 + total_cost[total_cost <= 1]**0.5
        # print(total_cost.mean())
        
        for obstacle in self.obstacles:

            integral = obstacle.integrate_one_shot_monte_carlo_circles(states[:, 0], states[:, 1])

            # Add integral cost to total cost
            total_cost += integral*10

        vx = torch.abs(states[:, 3])
        vy = torch.abs(states[:, 4])
        # total_cost += vx + vy

        # Check for every entry in the control vector if it is outside the limits
        to_large_a = torch.abs(control[:, 0]) > 2
        to_large_s = torch.abs(control[:, 1]) > 2

        # total_cost += to_large_a*1 + to_large_s*1
        # total_cost += control[:, 0]**2 + control[:, 1]**2

        return total_cost
    

@hydra.main(version_base=None, config_path="../conf", config_name="config_no_isaac")
def run_mppi(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    steps = 400
    trajectory = torch.zeros((steps, 2))

    total_costs = torch.zeros((steps))

    obstacles = [Obstacle(cfg, 5, 5)]
    obstacles = []
    objective = Objective(cfg, obstacles)
    prior = None
    planner = MPPIisaacPlanner(cfg, objective, steps, prior)

    for i in range(steps):

        print(i)
        control = planner.mppi.command(planner.current_state)
        objective.control = control
        planner.current_state = planner.forward_propagate(planner.current_state, control, cfg.mppi.dt)
        objective.current_state = planner.current_state
        total_costs[i] = planner.mppi.total_costs.mean()

        trajectory[i] = planner.current_state[:2]

    # Plot trajectory and obstacles
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    for obstacle in obstacles:
        position = obstacle.current_state.cpu().numpy()
        plt.plot(position[0], position[1], 'ro')
    plt.show()

    # Plot velocities over time
    plt.plot(total_costs.cpu().numpy())
    plt.plot(planner.x_dots.cpu().numpy())
    plt.plot(planner.y_dots.cpu().numpy())
    plt.plot(planner.theta_dots.cpu().numpy())
    plt.legend(['total_costs', 'x_dots', 'y_dots', 'theta_dots'])
    plt.show()


import cProfile
import pstats

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        run_mppi()
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(20)
        
