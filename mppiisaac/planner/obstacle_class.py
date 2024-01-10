import hydra
import time
import torch
import numpy as np

from omegaconf import OmegaConf
from scipy.integrate import nquad, dblquad
from scipy.stats import multivariate_normal
from multiprocessing import Pool, set_start_method
from typing import List

from mppiisaac.utils.config_store_no_isaac import ExampleConfig



class Obstacle(object):

    def __init__(self, cfg, x, y) -> None:

        self.cfg = cfg
        
        self.current_state = torch.zeros((self.cfg.nx), device=self.cfg.mppi.device)
        self.current_state[:2] = torch.tensor([x, y], device=self.cfg.mppi.device)

        self.cov = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=cfg.mppi.device)

        self.N_monte_carlo = 100000
        self.integral_radius = 1

        # TODO: Create a function that can be called to update the gaussian at a new timestep
        self.update_gaussian()


    def update_gaussian(self):

        # Create Gaussian used by the scipy merhod
        self.gaussian_scipy = multivariate_normal(mean=self.current_state.cpu()[:2], cov=self.cov.cpu())

        # Create Gaussian used by the torch and Monte Carlo method
        self.gaussian_torch = torch.distributions.multivariate_normal.MultivariateNormal(self.current_state[:2], self.cov)

        # Sample the torch Gaussian for Monte Carlo integration
        self.monte_carlo_sample()   


    ########## SCIPY VERSION ##########
    def integrand_scipy(self, x, y):
        point = np.array([x, y])
        return self.gaussian_scipy.pdf(point)

    def integrate_gaussian_scipy(self, x0, x1, y0, y1):
        # Define the limits for the integration
        limits = [(x0, x1), (y0, y1)]
        
        # Perform the integration using nquad and the wrapper function
        integral, _ = nquad(self.integrand_scipy, limits)
        return integral
    

    ########## MONTE CARLO VERSION FASTEST SO FAR ##########
    # Function that samples N points from the Gaussian distribution
    def monte_carlo_sample(self):

        # Create the samples
        self.samples = self.gaussian_torch.sample((self.N_monte_carlo,))

        # Sample a grid of points of shape (self.N_monte_carlo, 2) with x and y ranging from -5 to 5
        samples_x = torch.rand((self.N_monte_carlo), device=self.cfg.mppi.device) * 8*self.cov[0, 0] - 4*self.cov[0, 0]
        samples_y = torch.rand((self.N_monte_carlo), device=self.cfg.mppi.device) * 8*self.cov[1, 1] - 4*self.cov[1, 1]
        self.samples = torch.stack((samples_x, samples_y), dim=1)

        # Calculate the log probabilities of the samples and convert to probabilities
        log_probs = self.gaussian_torch.log_prob(self.samples)
        self.pdf_values = torch.exp(log_probs)


    # Calculate the integral of the gaussian over a rectangular area using Monte Carlo integration
    # This version takes in a single value for all bounds    
    def integrate_monte_carlo(self, x0, x1, y0, y1):
        
        # Check which samples are within the specified bounds
        within_bounds = ((self.samples[:, 0] >= x0) & (self.samples[:, 0] <= x1) &
                         (self.samples[:, 1] >= y0) & (self.samples[:, 1] <= y1))

        mean_within_bounds = torch.mean(self.pdf_values[within_bounds])


        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the rectangular region
        area = (x1 - x0) * (y1 - y0)
        integral_estimate = (mean_within_bounds) * area

        return integral_estimate


    # Calculate the integral of the gaussian over a rectangular area using Monte Carlo integration
    # This version takes in tensors for all bounds
    def integrate_one_shot_monte_carlo(self, x0, x1, y0, y1):

        x0, x1, y0, y1 = map(lambda v: torch.as_tensor(v, device=self.cfg.mppi.device), (x0, x1, y0, y1))

        # Check which samples are within the specified bounds
        within_x_bounds = (self.samples[:, 0, None] >= x0) & (self.samples[:, 0, None] <= x1)
        within_y_bounds = (self.samples[:, 1, None] >= y0) & (self.samples[:, 1, None] <= y1)
        within_bounds = within_x_bounds & within_y_bounds

        means_within_bounds = torch.zeros(len(x0), device=self.cfg.mppi.device)
        
        for i in range(len(x0)):
            means_within_bounds[i] = torch.mean(self.pdf_values[within_bounds[:, i]])
        
        # Replace all values with nan with 0
        means_within_bounds[means_within_bounds != means_within_bounds] = 0

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the rectangular region
        area = (x1 - x0) * (y1 - y0)
        integral_estimate = means_within_bounds * area

        return integral_estimate


    # Create a function that does the same as integrate_one_shot_monte_carlo but it takes in x and y which are the centers of circles 
    # The within bounds check has to be done on all samples if they are within a circle with radius r
    def integrate_one_shot_monte_carlo_circles(self, x, y):
            
        # Create the tensors for x, y and r
        x, y = map(lambda v: torch.as_tensor(v, device=self.cfg.mppi.device), (x, y))
        r = torch.ones((len(x)), device=self.cfg.mppi.device) * self.integral_radius

        # Check which samples are within the specified bounds
        within_bounds = ((self.samples[:, 0, None] - x)**2 + (self.samples[:, 1, None] - y)**2 <= r**2)

        means_within_bounds = torch.zeros(len(x), device=self.cfg.mppi.device)
        
        for i in range(len(x)):
            means_within_bounds[i] = torch.mean(self.pdf_values[within_bounds[:, i]])
        
        # Replace all values with nan with 0
        means_within_bounds[means_within_bounds != means_within_bounds] = 0

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the rectangular region
        area = torch.tensor(np.pi, device=self.cfg.mppi.device) * r**2
        integral_estimate = (means_within_bounds) * area

        return integral_estimate



    ########## TORCH VERSION MUCH SLOWER ##########
    def integrand_torch(self, x, y):
        point = torch.tensor([x, y], dtype=torch.float32, device=self.cfg.mppi.device)
        return np.exp(self.gaussian_torch.log_prob(point).item())
    
    def integrate_gaussian_torch(self, x0, x1, y0, y1):
        integral, _ = dblquad(self.integrand_torch, x0, x1, lambda x: y0, lambda x: y1)
        return integral
    
    # Function that can calculate the integral of a gaussian over a rectangular area in parallel using torch
    def integrate_gaussian_torch_parallel(self, x0, x1, y0, y1):                                                                                                                                                                                                  
        # Define the limits for the integration
        limits = [(x0, x1), (y0, y1)]
        
        # Perform the integration using nquad and the wrapper function
        integral, _ = nquad(self.integrand_torch, limits)
        return integral
        
    
    ########## FORWARD PROPAGAT ##########
    def forward_propagate(self, control, t):

        # Car parameters
        L = 2.0  # Length of the car (wheelbase)

        # Unpack state variables and control inputs
        x, y, yaw, x_dot, y_dot, yaw_dot = self.current_state

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


@hydra.main(version_base=None, config_path="../../conf", config_name="config_no_isaac")
def test_integral_speed(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    obstacle = Obstacle(cfg, 0, 0)
    N = 100

    calculate_scipy = True
    calculate_torch = False
    calculate_monte_carlo = True
    calculate_monte_carlo_one_shot = True
    calculate_monte_carlo_circles = True

    x0, x1, y0, y1 = -1, 1, -1, 1

    print(f"Calculating {N} integrals")

    if calculate_scipy:
        # Time calculation
        start = time.perf_counter()
        for _ in range(N):
            obstacle.integrate_gaussian_scipy(x0, x1, y0, y1)
        end = time.perf_counter()

        # print(f"Scipy result: {obstacle.integrate_gaussian_scipy(x0, x1, y0, y1)}")
        print(f"Time to integrate scipy: {end - start}")

    if calculate_torch:
        # Time calculation
        start = time.perf_counter()
        for _ in range(N):
            obstacle.integrate_gaussian_torch(x0, x1, y0, y1)
        end = time.perf_counter()

        # print(f"Torch result: {obstacle.integrate_gaussian_torch(x0, x1, y0, y1)}")
        print(f"Time to integrate torch: {end - start}")

    if calculate_monte_carlo:
        # Time calculation
        start = time.perf_counter()
        for _ in range(N):
            obstacle.integrate_monte_carlo(x0, x1, y0, y1)
        end = time.perf_counter()

        # print(f"Torch result: {obstacle.integrate_monte_carlo(x0, x1, y0, y1)}")
        print(f"Time to integrate monte carlo: {end - start}")

    if calculate_monte_carlo_one_shot:
        
        # Define the arrays for bounds
        x0 = -1*torch.ones(N, device=cfg.mppi.device)
        x1 = torch.ones(N, device=cfg.mppi.device)
        y0 = -1*torch.ones(N, device=cfg.mppi.device)
        y1 = torch.ones(N, device=cfg.mppi.device)

        # Time calculation
        start = time.perf_counter()
        integral_values = obstacle.integrate_one_shot_monte_carlo(x0, x1, y0, y1)
        end = time.perf_counter()

        # print(f"Parallel result: {integral_values[0]}")
        print(f"Time to integrate monte carlo one parallel: {end - start}")

    if calculate_monte_carlo_circles:

        # Sample a torch tensor of shape (N,) with x and y ranging from -5 to 5
        x = torch.rand((N), device=cfg.mppi.device) * 10 - 5
        y = torch.rand((N), device=cfg.mppi.device) * 10 - 5
        r = torch.ones((N), device=cfg.mppi.device)

        # Calculate the integral using monte carlo for circles
        start = time.perf_counter()
        monte_carlo_results = obstacle.integrate_one_shot_monte_carlo_circles(x, y)
        end = time.perf_counter()

        print(f"Time to integrate monte carlo circles: {end - start}")
    
    print()



@hydra.main(version_base=None, config_path="../../conf", config_name="config_no_isaac")
def test_integral_accuracy(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    obstacle = Obstacle(cfg, 0, 0)
    N = 1000

    scipy_results = []

    # Init empty lists for the bounds
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    print("The integral using the Monte Carlo approach compared to the scipy approach has")

    for _ in range(N):

        # Sample lower and upper bounds for x and y. Note: the lower bounds must be less than the upper bounds
        bound = 3
        x0_sample = np.random.uniform(-bound, bound)
        x1_sample = np.random.uniform(x0_sample, bound)
        y0_sample = np.random.uniform(-bound, bound)
        y1_sample = np.random.uniform(y0_sample, bound)

        # Calculate the integral using scipy
        result = obstacle.integrate_gaussian_scipy(x0_sample, x1_sample, y0_sample, y1_sample)

        # Append result to the list
        scipy_results.append(result)

        # Append the bounds to the lists
        x0.append(x0_sample)
        x1.append(x1_sample)
        y0.append(y0_sample)
        y1.append(y1_sample)


    monte_carlo_results = obstacle.integrate_one_shot_monte_carlo(x0, x1, y0, y1).cpu().numpy()
    scipy_results = np.array(scipy_results)

    # This is the error between the monte carlo and scipy results
    error = np.abs(monte_carlo_results - scipy_results)
    error = error[~np.isnan(error)]

    relative_error = np.abs(monte_carlo_results - scipy_results)/scipy_results
    relative_error = relative_error[~np.isnan(relative_error)]

    # Calculate mean of the error and its standard deviation
    mean = error.mean()
    std = error.std()

    print(f"An error of {mean} +/- {std}")
    print(f"And a relative error of {relative_error.mean()} +/- {relative_error.std()}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config_no_isaac")
def test_monte_carlo_circle(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    obstacle = Obstacle(cfg, 0, 0)
    N = 1000

    # Sample a torch tensor of shape (N,) with x and y ranging from -5 to 5
    x = torch.rand((N), device=cfg.mppi.device) * 10 - 5
    y = torch.rand((N), device=cfg.mppi.device) * 10 - 5

    # Calculate the integral using monte carlo for circles
    monte_carlo_results = obstacle.integrate_one_shot_monte_carlo_circles(x, y)

    # # This is the error between the monte carlo and scipy results
    # error = np.abs(monte_carlo_results - scipy_results)
    # error = error[~np.isnan(error)]

    # relative_error = np.abs(monte_carlo_results - scipy_results)/scipy_results
    # relative_error = relative_error[~np.isnan(relative_error)]

    # # Calculate mean of the error and its standard deviation
    # mean = error.mean()
    # std = error.std()

    # print("The integral using the Monte Carlo approach compared to the scipy approach has")
    # print(f"An error of {mean} +/- {std}")
    # print(f"And a relative error of {relative_error.mean()} +/- {relative_error.std()}")



if __name__ == "__main__":

    # Test the computational efficiency of multiple integral calculation methods
    test_integral_speed()

    # Test the accuracy of the integral calculation methods
    test_integral_accuracy()

    # Test the circles integral calculation method
    # test_monte_carlo_circle()

