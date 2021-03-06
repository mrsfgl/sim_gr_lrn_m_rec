# -*- coding: utf-8 -*-
import click
import logging
from omegaconf import OmegaConf
from pathlib import Path
from src.data.generate import *
from src.data.contaminate import *
from src.data.generate_graphs import *
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
    if input_params.type == 'stationary_smooth':
        Phi = generate_graphs(sizes = dims) # Add graph density as well.
        Y, V = generate_smooth_stationary_data(Phi)
    elif input_params.type == 'low_rank':
        ranks = input_params.ranks
        Y = generate_low_rank_data(dims, ranks)
    else:
        raise ValueError("Wrong data type.")

    Y_noisy = contaminate_signal(Y,
                                 noise_rate=input_params.noise_level, 
                                 noise_type=input_params.noise_type, 
                                 missing_ratio=input_params.missing_ratio) # Add noise type, noise parameter level, missing data percentage etc.

    return Y_noisy, Y, input_params # Consider saving the data rather than passing it along


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
