# %% [markdown]
# We test our tree explainer on the asthma exacerbation models.

# %%
# Load data and models
import joblib
from pathlib import Path

data_path = Path('..', 'datasets-and-models', 'models', 'data_00.joblib')
model_path = Path('..', 'datasets-and-models', 'models', 'model_00')

data = joblib.load(data_path)
model = joblib.load(model_path)


# %%
import numpy as np

False == 0

# %% [markdown]
# Data generator for the explanation tree
# 
# 

# %%
# Data generator.

import numpy as np
from generalizedtrees.constraints import vectorize_constraints

rng = np.random.default_rng(seed=20200508)

# Assuming independent
eps = 0.1

ndim = data['train_features'].shape[1]

# Estimate frequencies

frequency = min( max(np.mean(data['train_features']), eps), 1-eps)

def constrained_generator(n, constraints):
    upper, lower, upper_eq, lower_eq = vectorize_constraints(constraints, ndim)
    assert(all(upper_eq))
    assert(not any(lower_eq))

    result = rng.choice([True, False], size=n*ndim, p=[frequency, 1-frequency]).reshape((n, ndim))

    for d in range(ndim):
        if upper[d] == 0:
            result[:,d] = 0
        elif lower[d] == 0:
            result[:,d] = 1

    return result


# %%
# Learn tree
from generalizedtrees.trepanlike import make_trepanlike_classifier

Explainer = make_trepanlike_classifier(model, constrained_generator=constrained_generator)

explainer = Explainer()
explainer.build()

# %% Save explainer

joblib.dump(explainer, "the-explainer.joblib")
# %% [markdown]
# * Get RF performance on test set
# * Get explainer performance on test set
# * Get decision tree performance on test set
# * Get fidelity measurement
