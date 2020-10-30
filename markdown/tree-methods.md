# Existing tree-learning methods

There is a nontrivial variety of methods both about learning decision trees from data and learning trees from models.
Here I catalogue various methods and outline their unique contributions.

## Learning trees from data
TODO: CART, C4.5, Buntine 1991 - Bayesian trees
Maybe: Kwok and Carter 1990, Heath et al. 1993.

## Learning trees from models

### TREPAN
*Craven and Shavlik 1995*

[TODO]

### Born again trees
*Breiman and Shang 1996*

#### Sample generation: "smearing"

Given a training set $( \mathbf x^{(i)} | i \in 1:n )$ of $d$-dimensional instances,
to generate a new training sample $\mathbf x^{(+)}$:

1. Select random $a \in 1:n$
2. For each $j \in 1:d$,
    with probability $p_\mathrm{alt}$
    let $x^{(+)}_j \leftarrow x^{(a)}_j$, otherwise
    let $x^{(+)}_j \leftarrow x^{(b)}_j$ for random $b \in 1:n$.

#### Tree construction

When considering expansion of node $t$, new samples are generated until $N_s$ samples reach the node, and their labels are estimated by the oracle. Let the number of samples that needed to be generated be $N_g$.

Let $p_t = \frac{N_s}{N_g}$, and let $p_{y=j}$ be the proportion of instances of class $j$ in this sample.
The cost of node $t$ is then: $$\mathrm{cost}(t) =  (1-\max_j( p(j) ) )p_t$$

For regression $\mathrm{cost}(t) = \mathrm{var}(t)p_t$

Find the best split at $t$ and if either child has no original training set instances, then $t$ is terminal.

#### Pruning and subtree selection

As in CART.

#### Additional notes:

Tried a different data generation approach where components of the new instance are selected from its $k$ nearest neighbors, did not better than smearing.

### Distilling a Neural Network Into a Soft Decision Tree

_Frosst and Hinton (Google Brain Team) 2017_ [http://arxiv.org/abs/1711.09784]

#### Hierarchical Mixture of Bigots

For inner node $i$: the probability of sending $\mathbf x$ to the rightmost branch is
$$p_i( \mathbf x ) = \sigma( \mathbf x \mathbf w_i + \beta_i )$$
here, $\mathbf w_i$ is a 'learned filter' and $\sigma$ is the expit function.

For a leaf node $\ell$, the probability of predicting class $k$ is
$$Q^\ell_k = \frac{ \exp( \phi^\ell_k ) }{ \sum_{k'} \exp( \phi^\ell_{k'}) }$$

* A more explainable prediction is the one obtained by taking the prediction of the leaf with the maximum path probability
* A path-weighted average of leaves also makes sense, but the explanation is not as explainable.

#### Loss function
Let $T_k$ be the target distribution and P^\ell( \mathbf x) be the probability of arriving at leaf $\ell$.
Then the corss-entropy loss function is
$$L(\mathbf x ) = -\log \left( \sum_{\ell \in \text{leaves}} P^\ell( \mathbf x) T_k \log Q_k^\ell \right)$$

* The tree is learned by first deciding on the structure (size) and then training parameters.
* Note that the "splits" are now linear functions.
* Additional regularization is needed to drive the tree to have balanced mass distributions.

#### Important point
The black-box model (neural net) is used to provide the soft target distribution $T$.
There is no sample generation, but there is a weighted average over all training samples used in the training instead.

### Interpreting Blackbox Models via Model Extraction

_Bastani, Kim, and Bastani 2017_ [http://arxiv.org/abs/1705.08504]

#### Mathematical definitions

* $x \in \mathcal X \subseteq \mathbb R^d$
* Axis-aligned constraint: $C= (x_i \leq t)$ where $i \in [d] = \{1, \ldots, d\}$ and $t \in \mathbb R$.
* The feasible set of $C$ is $\mathcal F(C) = \{x \in \mathcal X | x \text{ satisfies } C\}$.
* A decision tree $T$ is a binary tree
    * An internal node $N = (N_L, N_R, C)$
    * A leaf node $N = (y); y\in \mathcal Y$
    * Root node $N_T$
    * A leaf node $(y)$ as a function $N(x) = y$
    * An internal node $(N_L, N_R, C)$ as a function
    $N(\mathbf x) =
    \begin{cases}
        N_L(x) & \text{if } x \in \mathcal F(C) \\
        N_R(x) & \text{Otherwise}
    \end{cases}$
    * The decision tree as a function $T(x) = N_T(x)$
    * $C_N$ is the conjunction of constraints along a path to $N$. Defined recursively:    
        * $C_T = \mathrm{True}$
        * Given $N = (N_L, N_R, C)$, $C_{N_L} = C_N \land C$ and $C_{N_R} = C_N \land \neg C$.

* Given black box $f: \mathcal X \rightarrow \mathcal Y$,
  Classification performance is held out fidelity, i.e.
  $$\frac{ 1 }{ | X_\text{test} | } \sum_{x \in X_\text{test} } \mathbb I [ T(x) = f(x) ]$$

#### Tree extractor

**Overview:** First use $X_\text{train}$ to estimate a distribution $\mathcal P$ over $\mathcal X$, then greedily learn $T$.

**Input distribution:**
Fit $X_\text{train}$ to a mixture of axis-aligned gaussians

* $\phi \in \mathbb R^k$
* $j \sim \mathrm{Categorical}(\phi)$
* $x \sim \mathcal N( \mu_j, \Sigma_j)$, $\Sigma_j$ diagonal

**Exact Greedy Decision Tree**

* Predetermined size $k$.
* Uses gini impurity to determine splits and decide on which node to split.
* The exact "infinite data" version is using the theoretical Gini impurity with respect to the probability distribution $\mathcal P$

**Estimated Greedy Decision Tree**

* Gini impurity now estimated by sampling from $\mathcal P$.
* They show how to sample from the mixture of axis-aligned gaussians with respect to axis-aligned constraints.
* They provide theoretical guarantees.