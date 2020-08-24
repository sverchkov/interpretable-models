# Draw ROCs of an explanation model

import joblib
import cloudpickle as cp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

def draw_roc_curves(y_true, predictions, out_file):
    
    fig, ax = plt.subplots()

    for label, y_score in predictions.items():

        fpr, tpr, _ = roc_curve(y_true, y_score)

        ax.plot(fpr, tpr, label=label)
    
    fig.legend()
    fig.savefig(out_file)


def get_p_of_1(model, data):

    p = model.predict_proba(data)
    if isinstance(p, pd.DataFrame):
        return p.iloc[:, 1]
    else:
        return p[:, 1]


def draw_agreement_scatter(models, reference, out_file):

    if reference not in models: raise ValueError("Can't find reference model in models")

    fig, ax = plt.subplots()

    x = models[reference]

    for key, y in models.items():
        if key != reference:
            ax.scatter(x, y, label=key)
    
    fig.legend()
    fig.savefig(out_file)


def cp_load(filename):
    with open(filename) as f:
        return cp.load(f)


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
        explanations = {
            label: cp_load(f) for label, f in
            zip(snakemake.params.labels, snakemake.input.explanations)}
        
        # Compute predictions
        predictions = {"RF": get_p_of_1(model, data["test_features"])}
        predictions.update({
            label: get_p_of_1(explanation, data["test_features"])
            for label, explanation in explanations.items()
        })

        # Draw ROC curves
        draw_roc_curves(data["test_targets"], predictions, out_file=snakemake.output.roc)

        # Draw agreement w/ explanation
        draw_agreement_scatter(
            predictions,
            reference = "RF",
            out_file = snakemake.output.agreement)
