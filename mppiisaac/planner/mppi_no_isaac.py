# from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
from mppiisaac.planner.mppi import MPPIPlanner
import mppiisaac
from typing import Callable, Optional
import io
import os
import yaml
from yaml.loader import SafeLoader

# from isaacgym import gymtorch
import torch

torch.set_printoptions(precision=2, sci_mode=False)


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)


class MPPIisaacPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, prior: Optional[Callable] = None):

        self.cfg = cfg
        self.objective = objective

        if prior:
            self.prior = lambda state, t: prior.compute_command(self.sim)
        else:
            self.prior = None

        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
        )

        # Note: place_holder variable to pass to mppi so it doesn't complain, while the real state is actually the isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))

        self.current_state = torch.zeros((1, self.cfg.nx), device=self.cfg.mppi.device)
    

    def update_objective(self, objective):
        self.objective = objective


    def dynamics(self, states, controls, t=0.1):
            
        # Car parameters
        L = 2.0  # Length of the car (wheelbase)

        # Unpack state variables and control inputs
        x, y, yaw, x_dot, y_dot, yaw_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        acceleration, steering_angle = controls[:, 0], controls[:, 1]

        # Update rates of change using the dynamic model
        yaw_dot += (x_dot / L) * torch.tan(steering_angle) * t
        x_dot += acceleration * torch.cos(yaw) * t
        y_dot += acceleration * torch.sin(yaw) * t

        # Update the state (position and orientation) based on the rates of change
        x += x_dot * t
        y += y_dot * t
        yaw += yaw_dot * t

        # Stack the updated state variables
        updated_states = torch.stack([x, y, yaw, x_dot, y_dot, yaw_dot], dim=1)

        return (updated_states, controls)
    

    def forward_propagate(self, state, control, t):

        # Car parameters
        L = 2.0  # Length of the car (wheelbase)

        # Unpack state variables and control inputs
        try:
            x, y, yaw, x_dot, y_dot, yaw_dot = state[0]
        except TypeError:
            x, y, yaw, x_dot, y_dot, yaw_dot = state

        acceleration, steering_angle = control

        # Update rates of change using the dynamic model
        yaw_dot += (x_dot / L) * torch.tan(steering_angle) * t
        x_dot += acceleration * torch.cos(yaw) * t
        y_dot += acceleration * torch.sin(yaw) * t

        # Update the state (position and orientation) based on the rates of change
        x += x_dot * t
        y += y_dot * t
        yaw += yaw_dot * t

        # Create the updated state tensor
        updated_state = torch.tensor([x, y, yaw, x_dot, y_dot, yaw_dot], device=self.cfg.mppi.device)

        return updated_state


    def running_cost(self, state):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the state is already saved and accesible in the simulator itself, so we ignore it and pass a handle to the simulator.
        return self.objective.compute_cost(state)


    def compute_action(self, state):

        actions = self.mppi.command(self.state_place_holder).cpu()
        return actions

