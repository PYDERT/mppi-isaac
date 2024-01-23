import torch


def differential_drive_dynamics(states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
        # x, y, theta, v_left, v_right, wheel_dist, wheel_radius, dt):
    """
    Forward propagate the state of a differential drive robot.

    Parameters:
    x (float): Current x position
    y (float): Current y position
    theta (float): Current orientation (in radians)
    v_left (float): Speed of the left wheel
    v_right (float): Speed of the right wheel
    wheel_dist (float): Distance between the wheels
    wheel_radius (float): Radius of the wheels
    dt (float): Time interval over which to propagate

    Returns:
    tuple: New state (x, y, theta)
    """

    # wheel_radius = 0.08
    # wheel_dist = 0.494
    dt = 0.05

    # Save the inputs as variables
    x, y, theta = states[:, 1], states[:, 0], states[:, 2]
    v, w = actions[:, 0], actions[:, 1]

    delta_x = torch.zeros_like(x)
    delta_y = torch.zeros_like(y)
    new_theta = theta + w * dt

    rotating = torch.abs(w) > 1e-6

    # For robots moving straight (omega is approximately zero)
    delta_x[~rotating] = v[~rotating] * torch.cos(theta[~rotating]) * dt
    delta_y[~rotating] = v[~rotating] * torch.sin(theta[~rotating]) * dt

    delta_x[rotating] = v[rotating]/w[rotating] * (torch.sin(new_theta[rotating]) - torch.sin(theta[rotating]))
    delta_y[rotating] = -v[rotating]/w[rotating] * (torch.cos(new_theta[rotating]) - torch.cos(theta[rotating]))

    # Replace all Nan values with 0 of delta_x and delta_y
    delta_x[delta_x != delta_x] = 0
    delta_y[delta_y != delta_y] = 0

    # # Update the current position and orientation
    new_x = x + delta_x
    new_y = y + delta_y
    new_states = torch.stack([new_x, new_y, new_theta], dim=1)

    return new_states