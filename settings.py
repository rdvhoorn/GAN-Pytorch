import logging

# TQDM_BAR_FORMAT helps to format the logging during the GAN training process.
TQDM_BAR_FORMAT = '[{elapsed}<{remaining},{rate_fmt}{postfix}]{percentage:3.0f}%|{bar:10}{desc}'

# Configure a logger without too much unnecessary information.
logging.basicConfig(format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
LOGGER = logging.getLogger()
