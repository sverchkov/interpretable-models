# Draw ROCs of an explanation model

import joblib
import cloudpickle as cp
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np

def draw_roc_curves(y_true, predictions, highlight, out_file):
    
    fig, ax = plt.subplots()

    for label, y_score in predictions.items():

        fpr, tpr, _ = roc_curve(y_true, y_score)

        auc = roc_auc_score(y_true, y_score)

        ax.plot(fpr, tpr, label=f'{label} AUC={auc:.4f}')

        # Highlighted points
        if label in highlight:
            t = highlight[label]
            predict_pos = y_score >= t
            fpr = (sum(np.logical_and(predict_pos, np.logical_not(y_true)))
                / sum(np.logical_not(y_true)))
            tpr = sum(np.logical_and(predict_pos, y_true)) / sum(y_true)
            ax.scatter(fpr, tpr, label=f'{label} at threshold {t:.4f}')
    
    fig.legend(loc='lower right')
    fig.savefig(out_file)


def get_p_of_1(model, data):

    p = model.predict_proba(data)
    if isinstance(p, pd.DataFrame):
        return p.iloc[:, 1]
    else:
        return p[:, 1]


def draw_agreement_scatter(models, y_true, reference, out_file):

    if reference not in models: raise ValueError("Can't find reference model in models")

    fig, ax = plt.subplots()

    x = models[reference]

    # [MarkerStyle(marker='o', fillstyle='full' if t else 'none') for t in y_true]
    #fills = ['full' if t else 'none' for t in y_true]

    for key, y in models.items():
        if key != reference:
            ax.scatter(x[y_true], y[y_true], alpha=0.6, label=f'{key} (+)')
            ax.scatter(x[np.logical_not(y_true)], y[np.logical_not(y_true)], alpha=0.6, label=f'{key} (-)')
    
    fig.legend(loc='center right')
    fig.savefig(out_file)


def cp_load(filename):
    with open(filename, 'rb') as f:
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

        highlight = {}
        try:
            with open(snakemake.input.rf_threshold, 'r') as f:
                highlight['RF'] = float(f.read())
        except:
            pass

        # Draw ROC curves
        draw_roc_curves(data["test_targets"], predictions, highlight, out_file=snakemake.output.roc)

        # Draw agreement w/ explanation
        draw_agreement_scatter(
            predictions,
            data['test_targets'],
            reference = "RF",
            out_file = snakemake.output.agreement)
