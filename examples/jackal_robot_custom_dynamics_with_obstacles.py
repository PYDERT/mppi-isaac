import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericDiffDriveRobot
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
from mppiisaac.utils.config_store import ExampleConfig
import time

from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mppiisaac.obstacles.obstacle_class import DynamicObstacles
from mppiisaac.dynamics.jackal_robot import jackal_robot_dynamics

# MPPI to navigate a simple robot to a goal position

# Set velocities of the obstacles. Not very nice but it works for the example
N_obstacles = 10  # Number of obstacles with maximum of 10
vx = torch.rand(N_obstacles, device="cuda:0") * 2 - 1
vy = torch.rand(N_obstacles, device="cuda:0") * 2 - 1
dt = 0.05  # Check this by printing env.dt() somewhere
cov_growth_factor = 1.0

# TODO: This example doesn't work yet as intended. Obstacles aren't avoided properly.
class Objective(object):
    def __init__(self, cfg, obstacles):

        self.cfg = cfg

        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.obstacles = obstacles

    def compute_cost(self, state: torch.Tensor, t: int):
        positions = state[:, 0:2]
        rot_cost = torch.abs(state[:,5])
        goal_dist = torch.linalg.norm(positions - self.nav_goal, axis=1)

        # If t is 0, we update the states of the obstacles
        random_vel_range = 0.0

        global vx, vy
        vx = vx + torch.rand(N_obstacles, device="cuda:0") * random_vel_range - random_vel_range / 2
        vy = vy + torch.rand(N_obstacles, device="cuda:0") * random_vel_range - random_vel_range / 2

        if t == 0:
            self.obstacles.state_coordinates[:, 0] += dt * vx
            self.obstacles.state_coordinates[:, 1] += dt * vy
            self.obstacles.state_cov = torch.tensor([[0.05, 0.0], [0.0, 0.05]], device=self.cfg.mppi.device)
            self.obstacles.coordinates = self.obstacles.state_coordinates
            self.obstacles.cov = self.obstacles.state_cov
            self.obstacles.create_gaussians(self.obstacles.coordinates[:, 0],
                                self.obstacles.coordinates[:, 1],
                                self.obstacles.cov)
        
        # Otherwise we are calculating the expected location of the obstacles for a rollout with increased covariance
        else:
            self.obstacles.coordinates[:, 0] += dt * vx
            self.obstacles.coordinates[:, 1] += dt * vy
            self.obstacles.cov *= cov_growth_factor
            self.obstacles.create_gaussians(self.obstacles.coordinates[:, 0],
                                self.obstacles.coordinates[:, 1],
                                self.obstacles.cov)
 
        # Calculate the cost of the obstacles
        total_obstacle_cost = self.obstacles.integrate_one_shot_monte_carlo_circles(positions[:, 0], positions[:, 1])

        return goal_dist * 1.0 + rot_cost * 0.05 + total_obstacle_cost * 50

class Dynamics(object):
    def __init__(self, cfg):
        self.dt = cfg.dt

    def simulate(self, states, control, t):
        new_states, control = jackal_robot_dynamics(states, control, self.dt)
        return (new_states, control)


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
    with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/jackal.yaml') as f:
        heijn_cfg = yaml.load(f, Loader=SafeLoader)
    urdf_file = f'{os.path.dirname(mppiisaac.__file__)}/../assets/urdf/' + heijn_cfg['urdf_file']
    robots = [
        GenericDiffDriveRobot(
            urdf=urdf_file,
            mode="vel",
            actuated_wheels=[
                "rear_right_wheel",
                "rear_left_wheel",
                "front_right_wheel",
                "front_left_wheel",
            ],
            castor_wheels=[],
            wheel_radius = 0.098,
            wheel_distance = 2 * 0.187795 + 0.08,
        ),
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
    for i in range(len(obstacles.coordinates)):

        # Specify the obstacle trajectories and radius
        dynamicObst1Dict = {
        "type": "sphere",
        "geometry": {"trajectory": [f"{float(obstacles.state_coordinates[i][0].cpu())} + {float(vx.cpu()[i])} * t", 
                                    f"{float(obstacles.state_coordinates[i][1].cpu())} + { float(vy.cpu()[i]) } * t", "0.1"], 
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
    planner = MPPICustomDynamicsPlanner(cfg, objective, dynamics.simulate)

    return planner

def init_obstacles(cfg):

    # Initialise the random obstacle locations
    init_area = 4.0
    init_bias = 1.0

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

    obstacles = DynamicObstacles(cfg, x, y, cov, integral_radius=0.3)

    return obstacles

@hydra.main(version_base=None, config_path="../conf", config_name="config_jackal_robot_custom_dynamics.yaml")
def run_jackal_robot(cfg: ExampleConfig):
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
        print(f"Time: {(time.time() - t)} s")
        
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_jackal_robot()
