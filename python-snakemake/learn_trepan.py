# Testing trepan on exacerbation data

from pathlib import Path
from time import perf_counter
import logging

import joblib
import cloudpickle as cp
import pandas as pd

from generalizedtrees import Trepan
from generalizedtrees.features import FeatureSpec

# Constants
logger = logging.getLogger()

def learn_trepan(data_path, model_path, features_path):
    
    # Load Data
    logger.info('Loading data and model')
    data = joblib.load(data_path)
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    # Build explainer
    logger.info('Initializing explainer class')
    explainer = Trepan(max_tree_size=20, use_m_of_n=False)

    logger.info('Unsing full training set')
    train_x = data['train_features']
        
    logger.info(f'Training set size: {train_x.shape[0]}')

    train_df = pd.DataFrame(train_x, columns=feature_names)

    logger.info('Learning explanation:')
    t0 = perf_counter()

    explainer.fit(train_df, model.predict_proba)

    t1 = perf_counter()
    logger.info(f'Learned explanation in {t1 - t0} seconds.')

    return explainer

def save_trepan(trepan, path):

    with Path(path).open('wb') as f:
        cp.dump(trepan, f)
    
    logger.info('Saved explanation')

if __name__ == '__main__':

    try:
        snakemake
    except NameError:
        print("This script should be called from snakemake.")
        raise
    else:

        logging.basicConfig(level=logging.DEBUG, filename=snakemake.log[0])

        model = learn_trepan(snakemake.input.data, snakemake.input.model, snakemake.input.feature_names)
        save_trepan(model, snakemake.output[0])