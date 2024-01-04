import torch
import matplotlib.pyplot as plt

def update_car_states(states, controls, dt=0.1):
    # Car parameters
    L = 2.0  # Length of the car (wheelbase)

    # Unpack state variables and control inputs
    x, y, yaw, x_dot, y_dot, yaw_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
    acceleration, steering_angle = controls[:, 0], controls[:, 1]

    # Update rates of change using the dynamic model
    yaw_dot += (x_dot / L) * torch.tan(steering_angle) * dt
    x_dot += acceleration * torch.cos(yaw) * dt
    y_dot += acceleration * torch.sin(yaw) * dt

    # Update the state (position and orientation) based on the rates of change
    x += x_dot * dt
    y += y_dot * dt
    yaw += yaw_dot * dt

    # Stack the updated state variables
    updated_states = torch.stack([x, y, yaw, x_dot, y_dot, yaw_dot], dim=1)

    return updated_states

def create_control_inputs(horizon, samples, control_inputs):

    input_velocity_tensor = torch.normal(mean=1, std=0.5, size=(horizon, samples))
    input_steering_tensor = torch.normal(mean=0, std=0.5, size=(horizon, samples))
    
    control_inputs_tensor = torch.stack([input_velocity_tensor, input_steering_tensor], dim=2)
    return control_inputs_tensor
    
if __name__ == '__main__':

    horizon = 10
    samples = 200
    nx = 6
    nu = 2

    car_states = torch.zeros((samples, nx))
    control_inputs = create_control_inputs(horizon, samples, nu)

    trajectories = torch.zeros((horizon, samples, 2))
    yaws = torch.zeros((horizon, samples))

    # Create rollout
    for i in range(horizon):

        car_states = update_car_states(car_states, control_inputs[i])

        trajectories[i, :] = car_states[:, :2]
        yaws[i, :] = car_states[:, 2]

    
    for i in range(samples):
        plt.plot(trajectories[:, i, 0], trajectories[:, i, 1]) 

    plt.show()

    print(yaws)

    


