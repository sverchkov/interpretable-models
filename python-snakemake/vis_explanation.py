# Visualize an explanation

import cloudpickle

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
        
        explanation_to_html(model, snakemake.output.figure)