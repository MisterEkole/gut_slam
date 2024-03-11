import torch

def light_spread_func(x, k):
    return torch.pow(torch.abs(x), k)

def calib_p_model(x, y, z, k, g_t, gamma):
    """Calculates the expected light intensity based on the photometric model.
       Optimized for PyTorch.

    Args:
        x, y, z (torch.Tensor): Coordinates of the 3D point.
        k, g_t, gamma (torch.Tensor): Calibration parameters. 

    Returns:
        torch.Tensor: The calculated light intensity.
    """

    mu = light_spread_func(z, k)
    fr_theta = 1 / torch.pi  # Lambertian BRDF

    # Optimized calculation
    center_to_pixel = torch.norm(torch.stack([x, y, z]), dim=0)
    xy_norm = torch.norm(torch.stack([x, y]), dim=0)
    theta = 2 * torch.acos(xy_norm / center_to_pixel) / torch.pi

    L = (mu / center_to_pixel) * fr_theta * torch.cos(theta) * g_t 
    return torch.pow(torch.abs(L), gamma)  

def cost_func(I, L, sigma=1e-3):
    diff = I - L
    huber_mask = torch.abs(diff) < sigma
    norm = torch.where(huber_mask, diff**2 / (2 * sigma), torch.abs(diff) + (sigma / 2))
    return norm

def reg_func(grad, sigma=1e-3):
    g = torch.exp(-torch.norm(grad))  
    huber_mask = torch.norm(grad) < sigma
    norm = torch.where(huber_mask, torch.norm(grad) ** 2 / (2 * sigma), torch.abs(grad) + (sigma / 2))
    return g * norm
