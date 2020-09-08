# Visualizing forest importances

import joblib
import numpy as np

if __name__ == '__main__':

    try:
        snakemake
    except NameError:
        from sys import stderr
        print("This script should be called from snakemake.", file=stderr)
        raise
    else:

        model = joblib.load(snakemake.input.model)
        feature_names = joblib.load(snakemake.input.feature_names)

        outfile = snakemake.output.importances

        importances = model.best_estimator_.feature_importances_

        indices = np.argsort(importances)[::-1]

        with open(outfile, 'wt') as f:

            print('Features, ordered by impurity importances:', file=f)

            for i in indices:
                if importances[i] > 0:
                    print(f'{feature_names[i]} ({importances[i]})', file=f)
            
            print(f'{sum(importances <= 0)} additional features had 0 importance.', file=f)
            
        