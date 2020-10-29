---
title: Learning trees to explain models
subtitle: Group Meeting
date: October 30, 2020
author: Yuriy Sverchkov
revealjs-url: https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.9.2
---

## Model explanation

* Highly accurate supervised learning models are often difficult to interpret
    * Deep networks
    * Random forests
    * Boosted models
    * Nonlinear SVMs

---

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

## Interpretable model: decision tree

* Internal nodes describe splits of the data space (define the *scope* of their children)

* Leaf nodes make predictions (often constant estimators, but can be models too)

## Explanation tree learning

* Given a model $f: \mathcal X \rightarrow \mathcal Y$ learn a decision tree with high fidelity to $f$
* Greedy algorithm:
    * _Start w/ a queue of one node (the root)_
    * _While queue is not empty and stopping criteria are not met:_
        * _Pop a node from the queue_
        * ___Generate data__ $(X, f(X))$ in the scope of the node_
        * ___Find a split__ that maximizes a __split score__ on the data_
        * _Generate chilren for the split and push into the queue_

## Question

<table>
<tr><td> Which </td>
<td> split space <br/> split score <br/> data generator <br/> leaf model </td>
<td> makes </td>
<td> high fidelity <br/> iterpretable </td>
<td> trees? </tr>
</table>

## The [`generalizedtrees`](https://github.com/Craven-Biostat-Lab/generalizedtrees) python package

* A joint framework for tree learning that allows swapping in different components that correspond to each design decision.

* ["Recipes"](https://github.com/sverchkov/generalizedtrees/blob/0d5caf8e9a163b7d3c12db3c578d801ceae9e53c/generalizedtrees/recipes.py) for making decision tree learners

## Comparing data generation approaches

* Trepan (Craven and Shavlik 1995)
    * Independent per-feature kernel density or empirical distribution
    * Distributions are re-estimated locally as the tree grows
    * A statistical test is used to determine whether to re-estimate the distribution

* Born-again trees (Breiman and Shang 1996)
    * 'Smearing' - taking a training instance and randomly swapping a random subset of its features with other instances.

## UCI Breast cancer wisconsin dataset

* Black box model: Random forest

## ROC curves

## Agreement plots

## Asthma exacerbation dataset

* Black box model: Random forest

## ROC curves

## Agreement plots


### Decision tree learning (from training data)

* Standard: greedily grow the tree
    * Scores: gini, entropy, variance, Bayesian
    * Algorithm:
        * _At each potential internal node, select a split that maximizes the score on the training data._
        * _Stop splitting according to some criteria._
            * Tree depth
            * Data scarecity
            * Degenerate score

### 

#### Problems

* Maximizing the score at a higher node means possibly suboptimal choices lower down
* Training data at each decision dwindles as the tree grows

## Explanation tree learning

* Given a model $f: \mathcal X \rightarrow \mathcal Y$ learn a decision tree with high fidelity to $f$
* Following a similar algorithm to standard decision tree learning:
    * Given a score function
    * Algorithm:
        * __At each potential internal node, select a split that maximizes the score on data $(X, f(X))$__
            * Given a generator for $X$, we solve the data scarecity issue.

----
#### Other approaches:
* Frosst and Hinton 2017: Learn splits and leaves by gradient descent for a fixed tree skeleton

## Decisions in learning an explaining decision tree

* Condition classes at internal nodes
* Classes of leaf nodes
* Space search strategy
* Local score function
* Unlabeled data generator
* Stopping criteria


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
