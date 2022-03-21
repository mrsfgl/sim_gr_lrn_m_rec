# -*- coding: utf-8 -*-
import click
import logging
from omegaconf import OmegaConf
from pathlib import Path
from generate import *
from contaminate import *
from generate_graphs import *
from dotenv import find_dotenv, load_dotenv

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(data_config_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    input_params = OmegaConf.load(data_config_path)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dims = input_params.dims
    Phi = generate_graphs(sizes = dims) # Add graph density as well.
    Y = generate_smooth_stationary_data(Phi)
    Y_noisy = contaminate_signal(Y) # Add noise type, noise parameter level, missing data percentage etc.

    return Y_noisy, input_params # Consider saving the data rather than passing it along


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
