# Testing trepan on exacerbation data

from pathlib import Path
from time import perf_counter
import logging

import joblib
import cloudpickle as cp
import numpy as np
import pandas as pd

from generalizedtrees import Trepan
from generalizedtrees.features import FeatureSpec

# Constants
logger = logging.getLogger()

def learn_trepan(data, model, feature_names):
    
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

    logger.info(f'Explanation:\n{explainer.show_tree()}')

    return explainer

def save_trepan(trepan, path):

    with Path(path).open('wb') as f:
        cp.dump(trepan, f)
    
    logger.info('Saved explanation')


def balance_sample(features, targets):

    majority = False
    n = sum(targets)

    if n > len(targets)/2:
        n = len(targets) - n
        majority = True
    
    # Select n from majority class
    idx = np.concatenate(
        [(targets != majority).nonzero()[0],
        np.random.default_rng().choice((targets == majority).nonzero()[0], size=n)])
    logger.debug(f'Selecting indices:\n{idx}')

    return features[idx, :], targets[idx]



if __name__ == '__main__':

    try:
        snakemake
    except NameError:
        from sys import stderr
        print("This script should be called from snakemake.", file=stderr)
        raise
    else:

        logging.basicConfig(level=logging.DEBUG, filename=snakemake.log[0])

        # Load Data
        logger.info('Loading data and model')

        data = joblib.load(snakemake.input.data)
        model = joblib.load(snakemake.input.model)
        feature_names = joblib.load(snakemake.input.feature_names)

        if snakemake.params.balance_training:
            logger.info('Sampling a balanced training set')
            data['train_features'], data['train_targets'] = balance_sample(data['train_features'], data['train_targets'])
        
        if snakemake.params.balance_testing:
            logger.info('Sampling a balanced testing set')
            data['test_features'], data['test_targets'] = balance_sample(data['test_features'], data['test_targets'])

        explanation = learn_trepan(data, model, feature_names)
        save_trepan(explanation, snakemake.output[0])