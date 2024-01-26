import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mppiisaac.planner.mppi_custom_dynamics import MPPICustomDynamicsPlanner
import mppiisaac
import hydra
import yaml
from yaml import SafeLoader
from omegaconf import OmegaConf
import os
import torch
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mppiisaac.utils.config_store import ExampleConfig

from mppiisaac.obstacles.obstacle_class import DynamicObstacles
from mppiisaac.obstacles.predict_linear_trajectory import predict_linear
from mppiisaac.dynamics.point_robot import omnidirectional_point_robot_dynamics
from mppiisaac.dynamics.boxer import differential_drive_dynamics

import time
import random

cov_growth_factor = None
N_obstacles = None
vx = None
vy = None

class Objective(object):
    def __init__(self, cfg, obstacles):

        self.cfg = cfg

        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.obstacles = obstacles

    def compute_cost_old(self, state: torch.Tensor, t: int):

        # Calculate the distance to the goal
        positions = state[:, 0:2]
        goal_dist = torch.linalg.norm(positions - self.nav_goal, axis=1)

        # # If t is 0, we update the states of the obstacles
        # random_vel_range = 0.0

        # vx = vx + torch.rand(N_obstacles, device="cuda:0") * random_vel_range - random_vel_range / 2
        # vy = vy + torch.rand(N_obstacles, device="cuda:0") * random_vel_range - random_vel_range / 2

        if t == 0:
            self.obstacles.state_coordinates[:, 0] += self.cfg.dt * vx
            self.obstacles.state_coordinates[:, 1] += self.cfg.dt * vy
            self.obstacles.state_cov = torch.tensor([[0.05, 0.0], [0.0, 0.05]], device=self.cfg.mppi.device)
            self.obstacles.coordinates = self.obstacles.state_coordinates
            self.obstacles.cov = self.obstacles.state_cov
            self.obstacles.create_gaussians(self.obstacles.coordinates[:, 0],
                                self.obstacles.coordinates[:, 1],
                                self.obstacles.cov)
        
        # Otherwise we are calculating the expected location of the obstacles for a rollout with increased covariance
        else:
            self.obstacles.coordinates[:, 0] += self.cfg.dt * vx
            self.obstacles.coordinates[:, 1] += self.cfg.dt * vy
            self.obstacles.cov *= cov_growth_factor
            self.obstacles.create_gaussians(self.obstacles.coordinates[:, 0],
                                self.obstacles.coordinates[:, 1],
                                self.obstacles.cov)
 
        # Calculate the cost of the obstacles
        total_obstacle_cost = self.obstacles.integrate_one_shot_monte_carlo_circles(positions[:, 0], positions[:, 1])

        return goal_dist * 1.0 + total_obstacle_cost * 50.0


    def compute_cost(self, state: torch.Tensor, t: int):

        # I have a tensor names state of shape (K, T, nx) and I want it to be of shape (T, K, nx)
        state = state.permute(1, 0, 2)

        # I now want to make state of shape (T*K, nx)
        state = state.reshape(-1, self.cfg.nx)

        # Calculate the distance to the goal
        positions = state[:, 0:2]
        goal_dist = torch.linalg.norm(positions - self.nav_goal, axis=1)

        coordinates = self.obstacles.predicted_coordinates
        covs = self.obstacles.predicted_covs

        coordinates = coordinates.reshape(-1, 2)
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        covs = covs.reshape(-1, 2, 2)

        self.obstacles.create_gaussians(x, y, covs)
 
        # Calculate the cost of the obstacles
        total_obstacle_cost = self.obstacles.integrate_one_shot_monte_carlo_circles(positions[:, 0], positions[:, 1])

        return goal_dist * 1.0 + total_obstacle_cost * 2.0

class Dynamics(object):
    def __init__(self, cfg):
        self.dt = cfg.dt

    def step_dynamics(self, states, control, t):
        new_states = omnidirectional_point_robot_dynamics(states, control, self.dt)
        # new_states = differential_drive_dynamics(states, control, self.dt)
        return (new_states, control)


class Predict(object):
    def __init__(self, cfg):
        self.cfg = cfg
        global cov_growth_factor
        cov_growth_factor = self.cov_growth_factor = 1.1

    def predict(self, coordinates, cov, vx, vy):
        predicted_coordinates, predicted_cov = predict_linear(self.cfg, coordinates, cov, vx, vy)
        return (predicted_coordinates, predicted_cov)


def initalize_environment(cfg, obstacles) -> UrdfEnv:
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/point_robot.yaml') as f:
        heijn_cfg = yaml.load(f, Loader=SafeLoader)
    urdf_file = f'{os.path.dirname(mppiisaac.__file__)}/../assets/urdf/' + heijn_cfg['urdf_file']
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=cfg.dt, robots=robots, render=cfg.render)
    # Set the initial position and velocity of the point mass.
    env.reset()
    goal_dict = {
        "weight": 1.0,
        "is_primary_goal": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "desired_position": cfg.goal,
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
    env.add_goal(goal)

    # For every coordinate pair, init an obstacle
    for i in range(len(obstacles.state_coordinates)):

        # Specify the obstacle trajectories and radius
        dynamicObst1Dict = {
        "type": "sphere",
        "geometry": {"trajectory": [f"{float(obstacles.state_coordinates[i][0].cpu())} + {float(obstacles.vx.cpu()[i])} * t", 
                                    f"{float(obstacles.state_coordinates[i][1].cpu())} + {float(obstacles.vy.cpu()[i]) } * t", "0.1"], 
                    "radius": 0.1},
        "movable": False,
        }

        # Add to the environment
        dynamicSphereObst1 = DynamicSphereObstacle(
            name="simpleSphere", content_dict=dynamicObst1Dict
        )
        env.add_obstacle(dynamicSphereObst1)

    return env


def set_planner(cfg, obstacles):
    """
    Initializes the mppi planner for the point robot.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    # urdf = "../assets/point_robot.urdf"
    objective = Objective(cfg, obstacles)
    dynamics = Dynamics(cfg)
    predictor = Predict(cfg)
    planner = MPPICustomDynamicsPlanner(cfg, objective, dynamics.step_dynamics, predictor)

    return planner


def init_obstacles(cfg):

    # Set velocities of the obstacles. Not very nice but it works for the example
    global vx, vy, N_obstacles
    N_obstacles = 10  # Number of obstacles with maximum of 20
    vx = torch.rand(N_obstacles, device="cuda:0") * 4 - 2
    vy = torch.rand(N_obstacles, device="cuda:0") * 4 - 2

    # Initialise the random obstacle locations
    init_area = 6.0
    init_bias = 2.0

    x = torch.rand(N_obstacles, device=cfg.mppi.device)*init_area-init_bias
    y = torch.rand(N_obstacles, device=cfg.mppi.device)*init_area-init_bias

    # Make sure the obstacles don't overlap with the robot
    for i in range(len(x)):
        if abs(x[i]) < 0.5 and abs(y[i]) < 0.5:
            x[i] += 1.0
            y[i] += 1.0

    # Create covariance matrix which consists of N_obstacles stacks of 2x2 covariance matrices
    cov = torch.tensor([[0.05, 0.0], [0.0, 0.05]], device=cfg.mppi.device)
    cov = cov.repeat(N_obstacles, 1, 1)

    obstacles = DynamicObstacles(cfg, x, y, cov, vx, vy, integral_radius=0.15)

    return obstacles


@hydra.main(version_base=None, config_path="../conf", config_name="config_point_robot_custom_dynamics.yaml")
def run_point_robot(cfg: ExampleConfig):
    """
    Set the gym environment, the planner and run point robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    obstacles = init_obstacles(cfg)
    env = initalize_environment(cfg, obstacles)
    planner = set_planner(cfg, obstacles)

    action = np.zeros(int(cfg.nx / 2))
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        t = time.time()
        ob_robot = ob["robot_0"]
        state = np.concatenate((ob_robot["joint_state"]["position"], ob_robot["joint_state"]["velocity"]))
        action = planner.compute_action(state)
        print("Action step took: ", time.time() - t)
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot()
