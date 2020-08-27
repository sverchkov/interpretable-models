---
title: Permutation-based feature importances can be misleading
subtitle: Craven Group meeting
author: Yuriy Sverchkov
date: August 28, 2020
revealjs-url: https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.9.2
---

### Please Stop Permuting Features: An Explanation and Alternatives
#### Giles Hooker and Lucas Mentch, 2019

> When features in the training set exhibit statistical dependence, permutation methods can be highly misleading when applied to the original model.

### Feature importance

Given a model $f : \mathcal X \rightarrow \mathcal Y$ where $\mathcal X$ is made up of multiple features,
quantify, for each feature $j$, its contribution to the prediction.

### Variable Importance

Breiman (2001)

$$VI_j^\pi = \sum_{i=1}^N L(y_i, f(\mathbf x_i^{\pi,j})) -  L(y_i, f(\mathbf x_i))$$

### Ground truth model

### Gaussian copula

### Predictions outside the training set

### Gibbs effects

(Neural network as a higher-order polynomial)