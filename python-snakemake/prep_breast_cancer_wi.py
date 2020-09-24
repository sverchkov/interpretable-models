# Visualizing forest importances

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer

if __name__ == '__main__':

    try:
        snakemake
    except NameError:
        from sys import stderr
        print('This script should be called from snakemake.', file=stderr)
        raise
    else:

        # Read output targets
        data_files = snakemake.output.datasets
        model_files = snakemake.output.models
        features_file = snakemake.output.feature_names

        # Light input verification
        n = len(data_files)
        assert n == len(model_files)

        # Load full dataset
        bc = load_breast_cancer()

        # Save feature names
        joblib.dump(bc['feature_names'], features_file)

        for i in range(n):

            # Make train-test split
            data = dict()
            (
                data['train_features'],
                data['test_features'],
                data['train_targets'],
                data['test_targets']
            ) = train_test_split(bc['data'], bc['target'])

            # Save train-test split
            joblib.dump(data, data_files[i])

            # Learn best rf model
            param_grid = {'max_depth': range(5,35,5)}
            estimator = RandomForestClassifier(n_estimators=100)
            model = GridSearchCV(estimator, param_grid, cv=5, iid=False, return_train_score=True,
                                scoring=make_scorer(roc_auc_score, needs_proba=True))
            model.fit(data['train_features'], data['train_targets'])
            
            # Save model
            joblib.dump(model, model_files[i])
