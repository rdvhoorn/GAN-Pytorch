import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential


class Generator(Module):
    """
    A basic implementation for a generator network.

    This generator implementation uses a standard multi-layer perceptron (MLP).
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            dropout: float = 0.0,
            leaky_relu_slope: float = 0.2,
    ):
        """
        The initialization function for the generator.

        :param input_size:          The input size of the generator.
                                        This should be equal to the size of the random prior.
        :param output_size:         The output size of the generator.
                                        This should be equal to the size of the feature space.
        :param dropout:             The dropout rate.
        :param leaky_relu_slope:    The slope of the leaky ReLU activation function.
        """
        super().__init__()

        # Initialize the sequential multi-layer perceptron model using the given parameters.
        self.sequential_model = Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Linear(in_features=256, out_features=512),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Linear(in_features=512, out_features=1024),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),

            nn.Linear(in_features=1024, out_features=output_size),
            # Final Tanh activation function to force all outputs to be in the range [-1, 1]
            nn.Tanh()
        )

    def forward(self, batch: Tensor) -> Tensor:
        """
        A function that generates fake samples for a batch of priors.

        :param batch:   A batch of priors (random tensors).
        :return:        A batch of fake samples.
        """
        return self.sequential_model(batch)
