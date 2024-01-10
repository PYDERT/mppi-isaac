import torch

class GaussianIntegrator:
    def __init__(self, current_state, cov, num_samples=10000):
        self.current_state = current_state
        self.cov = cov
        self.num_samples = num_samples
        self.gaussian_torch = torch.distributions.multivariate_normal.MultivariateNormal(self.current_state, self.cov)
    
    def monte_carlo_integration(self, x_lower, x_upper, y_lower, y_upper):
        # Generate samples from the Gaussian distribution
        samples = self.gaussian_torch.sample((self.num_samples,))
        
        # Check if the samples are within the bounds
        within_bounds_x = (samples[:, 0] >= x_lower) & (samples[:, 0] <= x_upper)
        within_bounds_y = (samples[:, 1] >= y_lower) & (samples[:, 1] <= y_upper)
        within_bounds = within_bounds_x & within_bounds_y
        
        # Calculate the proportion of samples that fall within the bounds
        proportion_within_bounds = within_bounds.float().mean(dim=0)
        
        # Estimate the integral as the proportion within bounds times the area of the bounds
        area = (x_upper - x_lower) * (y_upper - y_lower)
        integral_estimates = proportion_within_bounds * area
        
        return integral_estimates

# Example usage:
N = 10  # Suppose we have 10 sets of bounds
x_lower = torch.rand(N)
x_upper = x_lower + torch.rand(N)  # Ensure upper bound is greater than lower bound
y_lower = torch.rand(N)
y_upper = y_lower + torch.rand(N)  # Ensure upper bound is greater than lower bound
current_state = torch.tensor([0.0, 0.0])
cov = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

integrator = GaussianIntegrator(current_state, cov)
integral_values = integrator.monte_carlo_integration(x_lower, x_upper, y_lower, y_upper)
