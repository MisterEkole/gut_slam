import torch

def get_calibration(img):
    """
    Calculates calibration parameters.  

    Args:
        img (torch.Tensor): Image tensor for calibration on the specified device.

    Returns:
        tuple: A tuple containing the calibration parameters (k, g_t, gamma) 
               as PyTorch tensors on the same device as the input image.
    """

    k = torch.tensor(2.5, device=img.device)
    g_t = torch.tensor(2.0, device=img.device)
    gamma = torch.tensor(2.2, device=img.device)

    return k, g_t, gamma
