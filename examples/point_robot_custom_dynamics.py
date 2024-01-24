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
from mppiisaac.utils.config_store import ExampleConfig
import time
from mppiisaac.dynamics.point_robot import omnidirectional_point_robot_dynamics

import time


class Objective(object):
    def __init__(self, cfg):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    # NOTE: takes in t for consistency with the custom dynamics example with obstacles
    def compute_cost(self, state: torch.Tensor, t: int):
        # Calculate the distance to the goal
        positions = state[:, 0:2]
        goal_dist = torch.linalg.norm(positions - self.nav_goal, axis=1)
        return goal_dist * 1.0

class Dynamics(object):
    def __init__(self, cfg):
        self.dt = cfg.dt

    def step_dynamics(self, states, control, t):
        new_states = omnidirectional_point_robot_dynamics(states, control, self.dt)
        return (new_states, control)


def initalize_environment(cfg) -> UrdfEnv:
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
    return env


def set_planner(cfg):
    """
    Initializes the mppi planner for the point robot.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    # urdf = "../assets/point_robot.urdf"

    objective = Objective(cfg)
    dynamics = Dynamics(cfg)
    planner = MPPICustomDynamicsPlanner(cfg, objective, dynamics.step_dynamics)

    return planner


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
    env = initalize_environment(cfg)
    planner = set_planner(cfg)

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
