import hydra
import time
import torch
import numpy as np

from omegaconf import OmegaConf
from scipy.integrate import nquad, dblquad
from scipy.stats import multivariate_normal
from multiprocessing import Pool, set_start_method
from typing import List

from mppiisaac.utils.config_store import ExampleConfig


class DynamicObstacles(object):

    def __init__(self, cfg, x, y, cov, vx, vy, N_monte_carlo=50000, sample_bound=5, integral_radius=0.15) -> None:

        # Set meta parameters
        self.print_time = False  # Set to True to print the time it takes to perform certain operations
        self.use_batch_gaussian = True  # True is faster

        # Save x, y and cov in the correct format
        if isinstance(x, int) or isinstance(x, float):
            x = torch.tensor([x], device=cfg.mppi.device)
        
        if isinstance(y, int) or isinstance(y, float):
            y = torch.tensor([y], device=cfg.mppi.device)

        if cov.ndim == 2:
            cov = cov.unsqueeze(0)

        # Save the inputs as the actual current state of the obstacle
        self.cfg = cfg
        self.state_coordinates = torch.stack((x, y), dim=1)
        self.state_cov = cov
        self.vx = vx
        self.vy = vy
        self.N_obstacles = len(x)
        self.t = cfg.mppi.horizon

        # Initialise the predicted states of the obstacle
        self.predicted_coordinates = None
        self.predicted_covs = None

        # Set values used for monte carlo integration
        self.N_monte_carlo = N_monte_carlo
        self.integral_radius = integral_radius
        sample_bound = sample_bound
        self.map_x0 = self.map_y0 = -sample_bound
        self.map_x1 = self.map_y1 = sample_bound

        # Sample a grid of points of shape (self.N_monte_carlo, 2) with x and y ranging from -5 to 5
        samples_x = torch.rand((self.N_monte_carlo), device=self.cfg.mppi.device) * (self.map_x1 - self.map_x0) + self.map_x0
        samples_y = torch.rand((self.N_monte_carlo), device=self.cfg.mppi.device) * (self.map_y1 - self.map_y0) + self.map_y0
        self.samples = torch.stack((samples_x, samples_y), dim=1)

        # For the initialisation, create all Gaussians
        # self.create_gaussians(x, y, cov, use_only_batch_gaussian=False)
        # Save x, y and cov in the correct format
        if isinstance(x, int) or isinstance(x, float):
            x = torch.tensor([x], device=self.cfg.mppi.device)
        
        if isinstance(y, int) or isinstance(y, float):
            y = torch.tensor([y], device=self.cfg.mppi.device)

        if cov.ndim == 2:
            cov = cov.unsqueeze(0)

        # Set initial values for the gaussian
        self.coordinates = torch.stack((x, y), dim=1)
        self.cov = cov
    
    def update_velocties(self, vx, vy):
        self.vx = vx
        self.vy = vy

    def update_predicted_states(self, coordinates, covs):
        self.predicted_coordinates = coordinates
        self.predicted_covs = covs

    # Function that can be called to create all Gaussians or only the most efficient, batch Gaussians
    def create_gaussians(self, x, y, cov, use_only_batch_gaussian=True):

        # Save x, y and cov in the correct format
        if isinstance(x, int) or isinstance(x, float):
            x = torch.tensor([x], device=self.cfg.mppi.device)
        
        if isinstance(y, int) or isinstance(y, float):
            y = torch.tensor([y], device=self.cfg.mppi.device)

        if cov.ndim == 2:
            cov = cov.unsqueeze(0)

        # Set initial values for the gaussian
        self.coordinates = torch.stack((x, y), dim=1)
        self.cov = cov

        # Create only batch Gaussian
        if use_only_batch_gaussian:
            self.update_gaussian_batch(self.coordinates, self.cov)
        # Otherwise create all Gaussians
        else:
            self.update_gaussian_scipy(self.coordinates, self.cov)
            self.update_gaussian(self.coordinates, self.cov)
            self.update_gaussian_batch(self.coordinates, self.cov)


    ########## SCIPY VERSION ##########
        
    def update_gaussian_scipy(self, coordinates, cov):

        # Create Gaussian used by the scipy merhod
        coordinates = coordinates.cpu()[0]
        cov = cov.cpu()[0]

        # Create the Gaussian
        self.gaussian_scipy = multivariate_normal(mean=coordinates, cov=cov)

    def integrand_scipy(self, x, y):
        point = np.array([x, y])
        return self.gaussian_scipy.pdf(point)

    def integrate_gaussian_scipy(self, x0, x1, y0, y1):
        # Define the limits for the integration
        limits = [(x0, x1), (y0, y1)]
        
        # Perform the integration using nquad and the wrapper function
        integral, _ = nquad(self.integrand_scipy, limits)
        return integral


    ########## UPDATE TORCH GAUSSIANS ##########

    # Function that creates a list of torch Gaussians
    def update_gaussian(self, coordinates, cov):

        t = time.time()

        # Create Gaussian used by the torch and Monte Carlo method
        self.torch_gaussians = []
        for i in range(len(coordinates)):
            self.torch_gaussians.append(torch.distributions.multivariate_normal.MultivariateNormal(coordinates[i], cov[i]))
 
        if self.print_time:
            print(f"Time to create torch gaussians: {time.time() - t}")

        # Sample the torch Gaussian for Monte Carlo integration
        self.monte_carlo_sample()   

    # Function that samples N points from the Gaussian distributions
    def monte_carlo_sample(self):

        t = time.time()

        # Calculate the log probabilities of the samples and convert to probabilities
        self.pdf_values = torch.zeros((self.N_monte_carlo), device=self.cfg.mppi.device)

        for gaussian in self.torch_gaussians:
            log_probs = gaussian.log_prob(self.samples)
            self.pdf_values += torch.exp(log_probs)

        if self.print_time:
            print(f"Time to calculate pdf values: {time.time() - t}")


    ########## UPDATE TORCH GAUSSIANS BATCH ##########

    # Function that creates a single batch of torch Gaussians
    def update_gaussian_batch(self, coordinates, cov):

        # Create Gaussian batch used by the torch and Monte Carlo method
        t = time.time()

        self.torch_gaussian_batch = torch.distributions.multivariate_normal.MultivariateNormal(coordinates, cov)

        if self.print_time:
            print(f"Time to create torch gaussians batch: {time.time() - t}")

        # Sample the torch Gaussian for Monte Carlo integration
        self.monte_carlo_sample_batch()   

    def monte_carlo_sample_batch(self):

        t = time.time()

        # Expand points to match the batch size and compute log_prob
        expanded_points = self.samples.unsqueeze(1).expand(-1, self.torch_gaussian_batch.batch_shape[0], -1)
        log_probs = self.torch_gaussian_batch.log_prob(expanded_points)

        self.sum_pdf_batch = torch.zeros((self.N_monte_carlo, self.cfg.mppi.horizon), device=self.cfg.mppi.device)
        
        # Sum the exponentiated log probabilities
        for i in range(self.N_obstacles):
            self.sum_pdf_batch[:, i] = torch.exp(log_probs[:, i*self.t:(i+1)*self.t]).sum(dim=1)  # NOTE: HERE THE SLICING MUGHT BE INCORRECT
        
        # This was the version where the cost calculation was called every timestep
        self.sum_pdf = torch.exp(log_probs).sum(dim=1)
        # self.sum_pdf = torch.exp(log_probs).max(dim=1).values  # Take the max rather than sum over all obstacles

        if self.print_time:
            print(f"Time to calculate pdf values batch: {time.time() - t}")


    ########## MONTE CARLO VERSION ##########

    # Calculate the integral of the gaussian over a rectangular area using Monte Carlo integration
    # This version takes in a single value for all bounds    
    def integrate_monte_carlo(self, x0, x1, y0, y1):
        
        # Check which samples are within the specified bounds
        within_bounds = ((self.samples[:, 0] >= x0) & (self.samples[:, 0] <= x1) &
                         (self.samples[:, 1] >= y0) & (self.samples[:, 1] <= y1))

        if self.use_batch_gaussian:
            mean_within_bounds = torch.mean(self.sum_pdf[within_bounds])
        else:
            mean_within_bounds = torch.mean(self.pdf_values[within_bounds])

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the rectangular region
        area = (x1 - x0) * (y1 - y0)
        integral_estimate = (mean_within_bounds) * area

        return integral_estimate

    # Calculate the integral of the gaussian over a rectangular area using Monte Carlo integration
    # This version takes in tensors for all bounds
    def integrate_one_shot_monte_carlo(self, x0, x1, y0, y1):

        if not isinstance(x0, torch.Tensor):
            x0, x1, y0, y1 = map(lambda v: torch.as_tensor(v, device=self.cfg.mppi.device), (x0, x1, y0, y1))

        # Check which samples are within the specified bounds
        within_x_bounds = (self.samples[:, 0, None] >= x0) & (self.samples[:, 0, None] <= x1)
        within_y_bounds = (self.samples[:, 1, None] >= y0) & (self.samples[:, 1, None] <= y1)
        within_bounds = within_x_bounds & within_y_bounds

        # Mask the values of the pdf_values tensor with the within_bounds tensor and calculate the column sums

        if self.use_batch_gaussian:
            masked_values = self.sum_pdf[:, None] * within_bounds  # Change self.pdf_values to self.sum_pdf to use batch version
        else:
            masked_values = self.pdf_values[:, None] * within_bounds  # Change self.pdf_values to self.sum_pdf to use batch version

        column_sums = torch.sum(masked_values, dim=0)
        true_counts = torch.sum(within_bounds, dim=0)

        # Calculate the mean by dividing the sum by the count and avoid division by zero by using torch.where
        means_within_bounds = torch.where(true_counts > 0, column_sums / true_counts, torch.tensor(0.0))

        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the rectangular region
        area = (x1 - x0) * (y1 - y0)
        integral_estimate = means_within_bounds * area

        return integral_estimate

    # Create a function that does the same as integrate_one_shot_monte_carlo but it takes in x and y which are the centers of circles 
    # The within bounds check has to be done on all samples if they are within a circle with radius r
    def integrate_one_shot_monte_carlo_circles(self, x, y):
        
        # # Create the tensors for x, y and r
        # x, y = map(lambda v: torch.as_tensor(v, device=self.cfg.mppi.device), (x, y))
        # r = torch.ones((len(x)), device=self.cfg.mppi.device) * self.integral_radius

        # Check which samples are within the specified bounds
        within_bounds = ((self.samples[:, 0, None] - x)**2 + (self.samples[:, 1, None] - y)**2 <= self.integral_radius**2)

        if self.use_batch_gaussian:
            masked_values = self.sum_pdf[:, None] * within_bounds  # Change self.pdf_values to self.sum_pdf to use batch version
        else:
            masked_values = self.pdf_values[:, None] * within_bounds
            
        column_sums = torch.sum(masked_values, dim=0)
        true_counts = torch.sum(within_bounds, dim=0)

        means_within_bounds = torch.where(true_counts > 0, column_sums / true_counts, torch.tensor(0.0))
        
        # The integral is approximated as the proportion of points within the bounds multiplied by the area of the rectangular region
        integral_estimate = (means_within_bounds) * np.pi * self.integral_radius**2

        return integral_estimate


@hydra.main(version_base=None, config_path="../../conf", config_name="config_test_obstacle.yaml")
def test_integral_speed(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    # cfg = OmegaConf.to_object(cfg)

    x = torch.tensor([1.0, 0.5], device=cfg.mppi.device)
    y = torch.tensor([1.0, 1.0], device=cfg.mppi.device)
    cov = torch.tensor([[[0.3, 0.0], [0.0, 0.3]], [[0.3, 0.0], [0.0, 0.3]]], device=cfg.mppi.device)
    obstacle = DynamicObstacles(cfg, x, y, cov)
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

    # TODO: Add back the integrate_gaussian_torch function
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


@hydra.main(version_base=None, config_path="../../conf", config_name="config_test_obstacle.yaml")
def test_integral_accuracy(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    # cfg = OmegaConf.to_object(cfg)

    x = torch.tensor([0.0], device=cfg.mppi.device)
    y = torch.tensor([0.0], device=cfg.mppi.device)
    cov = torch.tensor([[0.3, 0.0], [0.0, 0.3]], device=cfg.mppi.device)

    x = 1
    y = 1

    obstacle = DynamicObstacles(cfg, x, y, cov)
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
        bound = 5
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
    print("These errors can be reduced by increasing N_monte_carlo in the obstacles class.")
    print("Also check the bounds of sampling and if only a single Guassian is integrated or more. Integrating 1 Gaussian only gives the actual error")




@hydra.main(version_base=None, config_path="../../conf", config_name="config_test_obstacle.yaml")
def test_monte_carlo_circle(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    # cfg = OmegaConf.to_object(cfg)

    obstacle = DynamicObstacles(cfg, 0, 0)
    N = 1000

    # Sample a torch tensor of shape (N,) with x and y ranging from -5 to 5
    x = torch.rand((N), device=cfg.mppi.device) * 10 - 5
    y = torch.rand((N), device=cfg.mppi.device) * 10 - 5

    # Calculate the integral using monte carlo for circles
    monte_carlo_results = obstacle.integrate_one_shot_monte_carlo_circles(x, y)


if __name__ == "__main__":

    # Test the computational efficiency of multiple integral calculation methods
    test_integral_speed()

    # Test the accuracy of the integral calculation methods
    test_integral_accuracy()

    # Test the circles integral calculation method
    # test_monte_carlo_circle()