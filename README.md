# Extrapolation in Feature Importance: Two Small Experiments

English | [中文](README_zh.md)

This repository contains two experiments on a common problem in explainable AI: feature-importance methods that evaluate a model at combinations of inputs that never occur in reality.

| Notebook | What it does |
|---|---|
| `shuffle_experiment.ipynb` | A toy marketing example of how unrestricted permutations inflate importance |
| `replicate_mr277.ipynb` | A replication of the Mr. 277 case (Table 2 and Figure 5) in Lu & Borgonovo (2026) using the authors' public code |

Notebook comments are in Chinese; the code and outputs are self-explanatory.

---

## Experiment 1: Unrestricted vs. within-group permutation

A small, self-contained experiment illustrating why **unrestricted permutations** can inflate feature importance when features are correlated, and how **restricting permutations to comparable observations** fixes the problem.

Inspired by Borgonovo, Cappelli, Lu, Plischke & Rudin, *Trustworthy Feature Importance Avoids Unrestricted Permutations* ([arXiv:2604.11253](https://arxiv.org/abs/2604.11253)).

### The idea in one paragraph

A common way to measure a variable's importance is to randomly shuffle its column and see how much the model's error increases (Breiman, 2001). When variables are correlated, shuffling one of them creates combinations that never occur in reality. The model has never seen such points, so its predictions there are unreliable (extrapolation), and the error increase gets wrongly attributed to the variable's importance.

### Setting

Simulated daily data for an online shop (1,000 days):

| Variable | Description |
|---|---|
| `holiday` | 1 if holiday (≈30% of days), 0 otherwise |
| `ad` | Ad spend: ≈80 on holidays, ≈20 on regular days (strongly tied to `holiday`) |
| `price` | Product price, ≈50 |
| `sales` | 250 + 150·holiday + 2·ad − 3·price + noise |

A random forest is trained to predict `sales`. We then measure the importance of `ad` in two ways:

1. **Free shuffle** – permute `ad` across all days.
2. **Within-group shuffle** – permute `ad` only among days of the same type (holiday with holiday, regular with regular).

### Results

| Method | Impossible rows created (out of 1,000) | Importance of `ad` (MSE increase) |
|---|---|---|
| Free shuffle | ~440 | ~8,900 |
| Within-group shuffle | ~0 | ~420 |

Same model, same data: the free shuffle reports an importance roughly **20× larger**. Almost all of the gap comes from the "impossible" rows (holidays with almost no ads, regular days with very high ads).

### How to run

Open `shuffle_experiment.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook) and run the cells from top to bottom. No local setup needed.

---

## Experiment 2: Replicating feasible Shapley values (Lu & Borgonovo, 2026)

Paper: X. Lu and E. Borgonovo, *How do I explain that decision? A new approach combining counterfactuals and sensitivity analyses*, European Journal of Operational Research 334(1), 200–215, 2026. [doi:10.1016/j.ejor.2026.04.035](https://doi.org/10.1016/j.ejor.2026.04.035)

Code and data: [LuXuefei/Feasible-Shapley-Values](https://github.com/LuXuefei/Feasible-Shapley-Values) (Apache 2.0). The notebook clones this repository and runs the German Credit case for applicant Mr. 277.

### Question

Mr. 277 is rejected for a loan (approval probability 0.07). A counterfactual suggests three changes that raise it to 0.51: shorten the loan duration from 60 to about 10 months, register a telephone, and hold at least 100 DM in savings. How much does each change contribute?

### Results

Of the 8 intermediate combinations, 2 fall below the density threshold (5th percentile of the training data) and are flagged as infeasible. Both shorten the loan to about 10 months without registering a telephone.

| Feature | Baseline Shapley (all combinations) | Feasible Shapley (infeasible points excluded) |
|---|---|---|
| Loan duration | 0.326 | 0.178 |
| Has telephone | 0.063 | 0.224 |
| Savings ≥ 100 | 0.051 | 0.039 |

All values, including the two infeasible points and their estimated densities (0.0145 and 0.0196), match the paper exactly. Excluding the infeasible points shifts the recommendation: registering a telephone becomes about as important as shortening the loan.

### Note on running the original code

The saved result files (`.pkl`) were created with an older version of pandas and fail to load with pandas 2.x (`No module named 'pandas.core.indexes.numeric'`). The notebook adds a small compatibility shim before loading them. The density model is loaded with `tf_keras` (legacy Keras 2), since Keras 3 does not read TensorFlow checkpoint files.

---

## Limitations and next steps

- Grouping by a single binary variable is the simplest form of restricted permutation. With continuous, multi-way correlations, the paper's approach (permuting residuals after a Gaussian transformation, GCMR) is needed.
- The data are simulated. A natural next step is to apply restricted-permutation importance to real marketing data (e.g., promotions, pricing and ad spend), where such correlations are common.
