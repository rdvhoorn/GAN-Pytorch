import torch
from torch import Tensor


def generate_2d_normal_prior(batch_size: int, prior_size: int) -> Tensor:
    """
    Function that generates a normally distributed prior of size (batch_size, prior_size).

    :param batch_size:  The batch size to generate.
    :param prior_size:  The number of dimensions to generate.
    :return:            A random prior of size (batch_size, z_dim).
    """
    return torch.randn(batch_size, prior_size)
