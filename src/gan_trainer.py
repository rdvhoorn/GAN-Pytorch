from typing import Callable, List, Iterator

import torch
from torch import Tensor
from torch.nn import Module, BCELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import generate_2d_normal_prior
from settings import TQDM_BAR_FORMAT, LOGGER


class GANTrainer:
    """
    This class implements and aggregates all the functionalities needed to train a generative adversarial network.

    There are 4 important functions:
    train:              trains the generative adversarial net for a given number of epochs.
    train_generator:    A function that trains the generator for one batch of data.
    """

    def __init__(
            self,
            generator: Module,
            discriminator: Module,
            generator_optimizer: Optimizer,
            discriminator_optimizer: Optimizer,
            dataloader: DataLoader,
            prior_size: int,
            prior_generation_function: Callable[[int, int], Tensor] = generate_2d_normal_prior,
            device: str = 'cpu'
    ):
        """
        :param generator:                   The generator object.
        :param discriminator:               The discriminator object. The discriminator should output in range [0,1] to
                                                be able to calculate the binary cross entropy loss.
        :param generator_optimizer:         The optimizer for the generator.
        :param discriminator_optimizer:     The optimizer for the discriminator.
        :param dataloader:                  The dataloader containing the data to train on.
        :param prior_size:                  Size of the prior. The prior is fed to the generator, so the input size
                                                of the generator should be equal to the size of the prior.
        :param prior_generation_function:   Function for generating a prior. Defaults to generating a standard normal
                                                distributed prior.
        :param device:                      The device on which the model is run.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.dataloader = dataloader
        self.prior_generation_function = prior_generation_function
        self.prior_size = prior_size
        self.device = device

        self.batch_size: int = dataloader.batch_size
        self.data = self.inf_batches()

        self.d_losses: List = []
        self.g_losses: List = []

        self.real_label = 1
        self.fake_label = 0

        self.bce_loss = BCELoss()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def train(self, n_epochs: int, n_generator: int, n_discriminator: int, verbose: bool = True) -> None:
        """
        A function to train the GAN for a given number of epochs.

        Training a GAN is done according to the following procedure:
        1. For n_epochs:
        2.      Train the discriminator for n_discriminator iterations
        3.      Train the generator for n_generator iterations
        4.      Store and log the losses

        :param n_epochs:            The number of epochs to train for.
        :param n_generator:         The number of iterations to train the generator for in one epoch.
        :param n_discriminator:     The number of iterations to train the discriminator for in one epoch.
        :param verbose:             Whether to print the losses of every epoch.
        """
        # Set generator and discriminator in training mode
        self.generator.train()
        self.discriminator.train()

        try:
            for epoch in (pbar := tqdm(range(1, n_epochs + 1),
                                       bar_format=TQDM_BAR_FORMAT, position=0, leave=True, disable=(not verbose))):
                d_losses_epoch, g_losses_epoch = [], []

                # 2. Train discriminator
                for _ in range(n_discriminator):
                    d_losses_epoch.append(self.train_discriminator())

                # 3. Train generator
                for _ in range(n_generator):
                    g_losses_epoch.append(self.train_generator())

                # 4. Store and log losses
                self.d_losses.append(float(torch.mean(torch.FloatTensor(d_losses_epoch))))
                self.g_losses.append(float(torch.mean(torch.FloatTensor(g_losses_epoch))))
                pbar.set_description_str('[%d/%d]: Generator loss: %.3f, Discriminator loss: %.3f' %
                                         (epoch, n_epochs, float(self.g_losses[-1]), float(self.d_losses[-1])))
        except KeyboardInterrupt:
            # If keyboard interrupt, simply let it pass.
            LOGGER.info("Successfully interrupted GAN training.")
            pass
        except Exception:
            # Raise other exceptions to next level.
            raise

    def train_generator(self) -> float:
        """
        A function that trains the generator for one batch of data.

        Training the generator is done according to the following procedure:
        1. Generate a prior
        2. Use the generator together with the prior to generate fake samples
        3. Discriminate those samples
        4. Calculate the loss
                If the generator is good, and is able to fool the discriminator, the loss is low
                If the generator is bad, and is not able to fool the discriminator, the loss is high
        5. Back propagate the loss
        6. Update the weights of the generator

        :return:        The generator loss.
        """
        # Reset gradients of generator optimizer
        self.generator_optimizer.zero_grad()

        # 1. Generate prior
        prior: Tensor = self.prior_generation_function(self.batch_size, self.prior_size).to(self.device)

        # 2, 3. Generate and discriminate data points
        generator_output: Tensor = self.generator(prior)
        discriminator_output: Tensor = self.discriminator(generator_output)

        # 4. Calculate the loss
        generator_loss: Tensor = self.bce_loss(discriminator_output,
                                               torch.full_like(discriminator_output, self.real_label))

        # 5, 6. Back propagate and update weights
        generator_loss.backward()
        self.generator_optimizer.step()

        # Return loss
        return generator_loss.data.item()

    def train_discriminator(self) -> float:
        """
        A function that trains the discriminator for one batch of data.

        Training the discriminator is done according to the following procedure:
        1. Obtain and discriminate a batch of real data
        2. Generate and discriminate a batch of fake data
                Generate a random prior
                Generate a batch of fake samples
                Discriminate the samples
        3. Calculate the discriminator loss
                Component 1 (real): If discriminator sees real data as real, component loss is low.
                Component 2 (fake): If discriminator sees fake data as fake, component loss is low.
                Losses are added up
        4. Back propagate the loss
        5. Update the weights of the discriminator

        :return:        The discriminator loss.
        """
        # Reset gradients of discriminator optimizer
        self.discriminator_optimizer.zero_grad()

        # 1. Get batch of data, and discriminate that data.
        batch_of_data = next(self.data)
        discriminator_output_real = self.discriminator(batch_of_data)

        # 2. Generate prior, generate samples using that prior, and then discriminate the generated samples
        prior = self.prior_generation_function(self.batch_size, self.prior_size).to(self.device)
        generator_output = self.generator(prior)
        discriminator_output_fake = self.discriminator(generator_output)

        # 3. Compute total loss
        discriminator_loss = self.bce_loss(discriminator_output_real,
                                           torch.full_like(discriminator_output_real, self.real_label)) + \
                             self.bce_loss(discriminator_output_fake,
                                           torch.full_like(discriminator_output_fake, self.fake_label))

        # 4, 5. Back propagate loss and update weights
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Return loss
        return discriminator_loss.data.item()

    def generate_n_samples(self, number_of_samples: int):
        """
        A function to generate n fake samples.

        :param number_of_samples:   The number of samples to generate.
        :return:                    The generated samples.
        """

        # Generate prior
        prior: Tensor = self.prior_generation_function(number_of_samples, self.prior_size).to(self.device)

        # Generate samples
        generator_output = self.generator(prior)

        # Return samples
        return generator_output

    def inf_batches(self) -> Iterator[Tensor]:
        """
        A function that returns an iterator (generator) object for batches of data.

        The function is called in the initialization function and assigned to `self.data`.
        By simply calling `next(self.data)`, a batch of data is obtained.
        You can keep calling `next(self.data)` infinitely.

        :return: A batch of data
        """
        while True:
            for batch_of_data in self.dataloader:
                yield batch_of_data.to(self.device)
