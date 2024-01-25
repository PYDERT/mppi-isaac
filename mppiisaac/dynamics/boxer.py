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

    wheel_radius = 0.08
    wheel_dist = 0.494
    dt = 0.05

    # Save the inputs as variables
    x, y, theta, vx, vy, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
    v, w = actions[:, 0], actions[:, 1]
    # v = torch.ones_like(v, device='cuda:0') * 0.6
    # w = torch.ones_like(v, device='cuda:0') * -0.8

    # def apply_base_velocity(self, vels: np.ndarray) -> None:
    #     """Applies forward and angular velocity to the base.

    #     The forward and angular velocity of the base
    #     is first transformed in angular velocities of
    #     the wheels using a simple dynamics model.

    #     """
    #     velocity_left_wheel = (
    #         vels[0] + 0.5 * self._wheel_distance * vels[1]
    #     ) / self._wheel_radius
    #     velocity_right_wheel = (
    #         vels[0] - 0.5 * self._wheel_distance * vels[1]
    #     ) / self._wheel_radius

    #     wheel_velocities = np.array([velocity_left_wheel, velocity_right_wheel])
    #     self.apply_velocity_action_wheels(wheel_velocities)

    v_left = (v - w * wheel_dist / 2)/wheel_radius
    v_right = (v + w * wheel_dist / 2)/wheel_radius

    delta_x = torch.zeros_like(x)
    delta_y = torch.zeros_like(y)
    new_theta = theta + w * dt

    rotating = torch.abs(w) > 1e-2

    # For robots moving straight (omega is approximately zero)
    delta_y[~rotating] = v[~rotating] * torch.cos(theta[~rotating]) * dt
    delta_x[~rotating] = v[~rotating] * torch.sin(theta[~rotating]) * dt

    radius = v[rotating]/w[rotating]
    delta_y[rotating] = radius * (torch.sin(new_theta[rotating]) - torch.sin(theta[rotating]))
    delta_x[rotating] = -radius * (torch.cos(new_theta[rotating]) - torch.cos(theta[rotating]))

    # Replace all Nan values with 0 of delta_x and delta_y
    delta_x[delta_x != delta_x] = 0
    delta_y[delta_y != delta_y] = 0

    # # Update the current position and orientation
    new_x = x + delta_x
    new_y = y + delta_y
    new_states = torch.stack([new_x, new_y, new_theta, vx, vy, omega], dim=1)

    return new_states