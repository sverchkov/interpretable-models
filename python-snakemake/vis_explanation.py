# Visualize an explanation

import cloudpickle
import pandas as pd

from generalizedtrees.vis import explanation_to_html

if __name__ == '__main__':

    try:
        snakemake
    except NameError:
        from sys import stderr
        print("This script should be called from snakemake.", file=stderr)
        raise
    else:

        with open(snakemake.input.model, 'rb') as f:
            model = cloudpickle.load(f)
        
        importances = None
        try:
            importances = pd.read_csv(snakemake.input.importances)
            
            importances['importance rank'] = importances.apply(
                lambda row: f"{row['permutation importance rank'] + 1}/{importances.shape[0]}",
                axis=1)
        except:
            print("Could not load feature importances file.")
        
        explanation_to_html(model, snakemake.output.figure, importances[['name', 'importance rank']])