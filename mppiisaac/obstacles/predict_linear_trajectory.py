import torch
import matplotlib.pyplot as plt

def predict_linear(cfg, coordinates, cov, vx, vy, plot=False):

    predicted_coordinates = torch.zeros((cfg.mppi.horizon, len(coordinates), 2), device=cfg.mppi.device)
    predicted_cov = torch.zeros((cfg.mppi.horizon, len(coordinates), 2, 2), device=cfg.mppi.device)

    for i in range(cfg.mppi.horizon):
        predicted_coordinates[i, :, 0] = coordinates[:, 0] = coordinates[:, 0] + cfg.dt * vx
        predicted_coordinates[i, :, 1] = coordinates[:, 1] = coordinates[:, 1] + cfg.dt * vy
        predicted_cov[i, :, :, :] = cov = cov * cfg.obstacles.cov_growth_factor
    
    if plot:
        for i in range(len(coordinates)):
            plt.plot(predicted_coordinates[:, i, 0].cpu(), predicted_coordinates[:, i, 1].cpu(), 'r')
        plt.grid()
        plt.show()
        plt.close()

    return predicted_coordinates, predicted_cov
