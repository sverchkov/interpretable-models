# Learning trees to explain models

_Group Meeting_

_May 8, 2020_

_Yuriy Sverchkov_

## Model explanation

* Highly accurate supervised learning models are often difficult to interpret
    * Deep networks
    * Random forests
    * Boosted models
    * Nonlinear SVMs
* There is a need in various settings to interpret model decisions
    * High-stakes decision making
        * Medical
        * Financial
        * Legal
    * Legal protections
    * User trust

## Post-hoc model-agnostic model translation

* __Post-hoc__: given a learned model $f: \mathcal X \rightarrow \mathcal Y$
* __model-agnostic__: without assumptions about the inner workings of the model
    * Contrast with saliency maps for CNNs
* __model translation__: we learn a model $g$ that performs like $f$ and is interpretable
    * Also called mimic learning

## Decision trees: our interpretable model of choice

* Pros:
    * Encode decision logic transparently
    * Cover the entire feature space by design
* Cons:
    * Relatively poor classifiers/regressors
    * Accurate trees tend to be deep

### Anatomy of a decision tree: internal nodes

Internal nodes represent conditions on features

* Axis-aligned splits (most common):
    * Thresholds for continuous/ordinal features
    * One-or-rest for discrete features
    * Every-value split for discrete features

* Composite condition splits:
    * $m$-of-$n$ conditions
    * Linear function splits

* Future ideas:
    * Interval segmentation
    * Latent/derived features
    * Splits for temporal data (e.g. was $x_i > \theta$ at some $t < \tau$)

### Anatomy of a decision tree: leaf nodes

Leaf nodes represent decisions

* Fixed-value prediction
* Fixed distribution
* Simple model ('model trees')

### Decision tree learning (from training data)

* Standard: greedily grow the tree
    * Scores: gini, entropy, variance, Bayesian
    * Algorithm:
        * _At each potential internal node, select a split that maximizes the score on the training data._
        * _Stop splitting according to some criteria._
            * Tree depth
            * Data scarecity
            * Degenerate score
    * Problems
        * Maximizing the score at a higher node means possibly suboptimal choices lower down
        * Training data at each decision dwindles as the tree grows

## Explanation tree learning

* Given a model $f: \mathcal X \rightarrow \mathcal Y$ learn a decision tree with high fidelity to $f$
* Following a similar algorithm to standard decision tree learning:
    * Given a score function
    * Algorithm:
        * __At each potential internal node, select a split that maximizes the score on data $(X, f(X))$__
            * Given a generator for $X$, we solve the data scarecity issue.

* Other approaches:
    * Frosst and Hinton 2017: Learn splits and leaves by gradient descent for a fixed tree skeleton

## Decisions in learning an explaining decision tree

* Condition classes at internal nodes
* Classes of leaf nodes
* Space search strategy
* Local score function
* Unlabeled data generator
* Stopping criteria

## The `generalizedtrees` python package

* [https://github.com/Craven-Biostat-Lab/generalizedtrees]
* Python package that implements a joint framework for all variants of tree learning and allows swapping in different components that correspond to each design decision.
* Already implemented:
    * Standard decision tree learning (verified against Scikit-Learn implementation)
    * Basic model-tree learning
    * Basic explanation (model translation) tree learning
* Compatible with Scikit-Learn

### `generalizedtrees` details: explanation tree learner

* Input and parameters:
    * Black-box classifier
    * Data generator
    * Impurity score function
    * Sample size to use for learning splits
    * Depth limit

```{python}
Explainer = make_trepanlike_classifier(classifier, generator)
ex_tree = Explainer(s_min, max_depth, score)
ex_tree.build()
```

## Planned evaluations

* Datasets: Asthma exacerbations, MIMIC-III, UW-Health OMOP CDM extract, others
* Sweeping comparison of many variants of explanation tree learning
* Related past work
    * Craven and Shavlik 1995
    * Breiman and Shang 1996
    * Bastani, Kim, and Bastani 2017
    * Frosst and Hinton 2017

### Trepan (Craven and Shavlik 1995)

* Data generation: Independent per-feature kernel density or empirical distribution
  * Distributions are re-estimated locally as the tree grows
  * A statistical test is used to determine whether to re-estimate the distribution
* Splits: $m$-of-$n$
* Stopping criteria: Statistical tests

### Born-again Trees (Breiman and Shang 1996)

* Data generation: 'Smearing' - taking a training instance and randomly swapping a random subset of its features with other instances.
* Data is generated first, then filtered by the tree
  * Rejection rate informs node scores (nodes that less samples reach score lower)
  * Accepted samples used to compute impurity at node
* Stopping criteria: Exhaustion of original training data at a node.

### Bastani, Kim, and Bastani 2017

* Data generation: mixture of gaussians.
* Efficient sampling subject to constraints.
* Only generated data is used.

### Frosst and Hinton 2017

* Internal nodes are logistic regression classifiers
* Leaf nodes are softmax functions
* Learning: tree structure is fixed, parameters (LR weights and biases, as well as softmax inputs) are learned by gradient descent

## Future developments

* Using model trees for explanation
* Splitting on higher-level concepts (feature groups, semantically meaningful latent features)
* Trees for temporal data (e.g. $x_i > \theta$ at $t < \tau$)
* Alternate scores for computing splits
