from mppiisaac.planner.mppi_no_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import torch
import zerorpc
import pytorch3d.transforms

from mppiisaac.utils.config_store import ExampleConfig

import matplotlib.pyplot as plt

class Objective(object):
    def __init__(self, cfg):
        
        # Old Code
        self.isaac = False
        if self.isaac:
            # Tuning of the weights for box
            self.w_robot_to_block_pos=  .2
            self.w_block_to_goal_pos=   2.
            self.w_block_to_goal_ort=   3.0
            self.w_push_align=          0.6
            self.w_collision=           10
            self.w_vel=                 0.

            # Task configration for comparison with baselines
            self.ee_index = 4
            self.block_index = 1
            self.ort_goal_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)

            self.block_goal_box = torch.tensor([0., 0., 0.5, 0.0, 0.0, 0.0, 1.0], device=cfg.mppi.device)
            self.block_goal_sphere = torch.tensor([0.42, 1., 0.5, 0, 0, -0.7071068, 0.7071068], device=cfg.mppi.device) # Rotation 90 deg

            # Select goal according to test
            self.block_goal_pose = torch.clone(self.block_goal_box)
            self.block_ort_goal = torch.clone(self.block_goal_pose[3:7])
            self.goal_yaw = torch.atan2(2.0 * (self.block_ort_goal[-1] * self.block_ort_goal[2] + self.block_ort_goal[0] * self.block_ort_goal[1]), self.block_ort_goal[-1] * self.block_ort_goal[-1] + self.block_ort_goal[0] * self.block_ort_goal[0] - self.block_ort_goal[1] * self.block_ort_goal[1] - self.block_ort_goal[2] * self.block_ort_goal[2])

            # Number of obstacles
            self.obst_number = 3        # By convention, obstacles are the last actors

        # New code
        self.goal_euler = torch.tensor(cfg.goal[:2], device=cfg.mppi.device)
        self.goal_orientation = torch.tensor([cfg.goal[2]], device=cfg.mppi.device)

        self.success = False
        self.count = 0

    
    def compute_cost(self, states):
        # Distances robot
        robot_to_goal = torch.linalg.norm(self.goal_euler - states[:, :2], dim=1, keepdim=True).squeeze(1)
        dyaw = 0.1*(states[:, 2] - self.goal_orientation)
        
        total_cost = robot_to_goal # + dyaw

        return total_cost
    

@hydra.main(version_base=None, config_path="../conf", config_name="config_no_isaac")
def run_mppi(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg)
    prior = None
    planner = MPPIisaacPlanner(cfg, objective, prior)

    steps = 300
    trajectory = torch.zeros((steps, 2))

    for i in range(steps):

        control = planner.mppi.command(planner.current_state)
        planner.current_state = planner.forward_propagate(planner.current_state, control, cfg.mppi.dt)

        trajectory[i] = planner.current_state[:2]

    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.show()


if __name__ == "__main__":
    run_mppi()
