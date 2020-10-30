# Visualizing forest importances

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

if __name__ == '__main__':

    try:
        snakemake
    except NameError:
        from sys import stderr
        print("This script should be called from snakemake.", file=stderr)
        raise
    else:

        model = joblib.load(snakemake.input.model)
        data = joblib.load(snakemake.input.data)
        feature_names = joblib.load(snakemake.input.feature_names)

        outfile = snakemake.output.importances

        clf = model.best_estimator_

        forest_importances = clf.feature_importances_

        # TODO: Make number of repeats a parameter
        repeats = 10
        if data['test_features'].shape[0] > 10000:
            repeats = 1

        pi = permutation_importance(
            clf,
            data['test_features'],
            data['test_targets'],
            n_repeats=10)

        importance_frame = pd.DataFrame({
            'name': feature_names,
            'permutation importance mean': pi.importances_mean,
            'permutation importance sd': pi.importances_std,
            'RF importance': forest_importances})
        
        importance_frame['permutation importance rank'] = (-importance_frame['permutation importance mean']).argsort()
        importance_frame['RF importance rank'] = (-importance_frame['RF importance']).argsort()

        importance_frame.to_csv(outfile)
            
        