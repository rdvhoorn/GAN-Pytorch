import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential

from src.utils import GANType


class Discriminator(Module):
    """
    A basic implementation for a discriminator network.

    This discriminator implementation uses a standard multi-layer perceptron (MLP).
    It can be used for WGAN and WGAN-GP training too, by specifying the discriminator_type as
    GANType.WGAN or GANType.WGAN_GP. The only difference this makes is that the final output
    tensor is turned into a probability distribution using the sigmoid activation function.
    """

    def __init__(
            self,
            input_size: int,
            dropout: float = 0.2,
            leaky_relu_slope: float = 0.2,
            discriminator_type: GANType = GANType.GAN
    ):
        """
        The initialization function for the discriminator.

        :param input_size:              The input size of the discriminator.
                                            This should be equal to the size of the feature space.
        :param dropout:                 The dropout rate.
        :param leaky_relu_slope:        The slope of the leaky ReLU activation function.
        :param discriminator_type:      What type of discriminator should be instantiated.
        """
        super().__init__()

        self.discriminator_type = discriminator_type

        # Initialize the sequential multi-layer perceptron model using the given parameters.
        self.sequential_model = Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, batch: Tensor) -> Tensor:
        """
        A function that computes the output of the discriminator for one batch of data.

        :param batch:       The batch (2d Tensor) to compute the output for
        :return:            The output (2d Tensor)
        """
        if self.discriminator_type == GANType.GAN:
            return torch.sigmoid(self.sequential_model(batch))
        else:
            return self.sequential_model(batch)
