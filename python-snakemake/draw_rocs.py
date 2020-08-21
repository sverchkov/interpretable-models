# Draw ROCs of an explanation model

import joblib
import cloudpickle as cp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

def draw_roc_curves(y_true, data, *models, out_file):
    
    fig, ax = plt.subplots()

    for model in models:

        predictions = model.predict_proba(data)
        if isinstance(predictions, pd.DataFrame):
            y_score = predictions.iloc[:,1]
        else:
            y_score = predictions[:,1]

        fpr, tpr, _ = roc_curve(y_true, y_score)

        ax.plot(fpr, tpr, label=str(type(model)))
    
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
        with open(snakemake.input.explanation, 'rb') as f:
            explanation = cp.load(f)
        
        # Compute predictions
        predictions = {
            "RF": get_p_of_1(model, data["test_features"]),
            "Trepan": get_p_of_1(explanation, data["test_features"])
        }

        # Draw ROC curves
        draw_roc_curves(data["test_targets"], data["test_features"], model, explanation, out_file=snakemake.output.figure)

        # Draw agreement w/ explanation
        draw_agreement_scatter(
            predictions,
            reference = "RF",
            out_file = snakemake.output.agreement)
