# Understanding Generative Adversarial Networks

This repository has the following purposes
1. Provide an [in-depth tutorial on how GANs work](https://github.com/rdvhoorn/GAN-Understandable-AI/blob/main/A-Z%20GAN%20tutorial.ipynb), including their theory, implementation, limitations, and advancements.
2. Provide the [most readable and intuitive implementation of the GAN framework ever written in Python](https://github.com/rdvhoorn/GAN-Understandable-AI/tree/main/src) consisting of 
    - A [Generator](https://github.com/rdvhoorn/GAN-Understandable-AI/blob/main/src/generator.py) object
    - A [Discriminator](https://github.com/rdvhoorn/GAN-Understandable-AI/blob/main/src/discriminator.py) object
    - A [Trainer class](https://github.com/rdvhoorn/GAN-Understandable-AI/blob/main/src/gan_trainer.py)

The implementation features the most basic approach to a GAN, using multi-layer perceptrons. However, the results
on the MNIST numbers dataset still show respectable quality:

![Results from basic training (GAN src example)](https://github.com/rdvhoorn/GAN-Understandable-AI/blob/main/images/results_plot.png)

Please use anything within this repository freely. Mentions are appreciated. In addition, you can contribute by opening pull-requests with improvements. 