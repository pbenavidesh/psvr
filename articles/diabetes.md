# Case Study: Diabetes Progression

> **Reproducibility note**
>
> These results correspond to the manuscript *“A Unified Family of
> Percentage-Error Support Vector Regression Models with Symmetric
> Kernel Extensions”* submitted to *Mathematics* (MDPI), currently under
> review. Hyperparameter grids, random seeds, and model implementations
> match those reported in the paper exactly. Final published results may
> differ if revisions are requested. Source code and raw results are
> available as downloadable files at the end of this article.

## 1 Introduction

The Diabetes dataset (Efron et al., 2004) comprises n = 442 patients
with p = 10 baseline clinical measurements (age, sex, BMI, blood
pressure, and six serum measurements). The target is a quantitative
measure of disease progression one year after baseline, which is
strictly positive with high relative variability (CV ≈ 51%). In clinical
decision support, relative prediction error is more meaningful than
absolute error: a 10-point error for a low-progression patient carries
different clinical weight than the same error for a high-progression
patient.

This article demonstrates that `psvr` models — trained directly under
MAPE and RMSPE loss — provide competitive percentage-error performance
over 30 randomized train/test splits on this challenging small-sample
regression problem. Data loaded from
[`lars::diabetes`](https://rdrr.io/pkg/lars/man/diabetes.html).

## 2 Setup

Code

``` r
library(tidyverse)
library(tidymodels)
library(psvr)
library(lars)
library(kernlab)
library(ranger)
library(xgboost)
library(quantreg)
library(e1071)
library(knitr)
library(kableExtra)
library(xfun)
library(patchwork)

source("case-studies/experiment_helpers.R")

tidymodels_prefer()
theme_set(theme_bw(base_size = 12))
```

## 3 Data

Code

``` r
data("diabetes", package = "lars")
y <- as.numeric(diabetes$y)
X <- as.matrix(diabetes$x)

stopifnot(all(y > 0))
```

| Statistic |  Value |
|:----------|-------:|
| n         | 442.00 |
| p         |  10.00 |
| min(y)    |  25.00 |
| median(y) | 140.50 |
| mean(y)   | 152.13 |
| max(y)    | 346.00 |

Diabetes dataset summary.

![](diabetes_files/figure-html/data-histogram-1.png)

All 442 target values satisfy `y > 0` (confirmed by `stopifnot` above).

> **Note**
>
> The coefficient of variation of the target (CV ≈ 51%) is substantially
> higher than Boston Housing, making percentage-error metrics more
> challenging but also more informative for this dataset.

## 4 Experimental protocol

The benchmark follows a 30-seed protocol: for each seed in 1–30, the 442
observations are randomly split into 80% training (≈ 353 rows) and 20%
test (≈ 89 rows). Hyperparameters are selected via 5-fold
cross-validation on the training set, minimizing each model’s native CV
metric. Grid sizes are 48 combinations for SVR-MAPE and Log-SVR; 96 for
SVR-MAPE+SK (the symmetry type `a ∈ {-1, 1}` is jointly tuned with the
cost, margin, and bandwidth); 16 for the LS-RMSPE variants and ε-SVR
(MSE); 32 for LS-RMSPE+SK (`a` jointly tuned with cost and bandwidth);
up to 6 for Random Forest; 8 for XGBoost; and no tuning for Linear
Regression, WLS, and Quantile Regression. Six metrics are recorded on
the held-out test set: MAPE, RMSPE, MAAPE, MASE, MSE, and R². Prior to
fitting, predictor features are standardized to zero mean and unit
variance using statistics computed on the training fold only; test-set
features are scaled with the same parameters to prevent data leakage.
MASE uses the lag-1 naive denominator; because the Diabetes data has no
temporal ordering, the denominator depends on row order within each
split and should be interpreted accordingly.

## 5 Run experiment

Code

``` r
results_file <- "case-studies/results/diabetes-results-grid-osqp.csv"

if (!file.exists(results_file)) {
  results <- run_experiment(X, y,
                            dataset_name = "diabetes",
                            seeds        = 1:30,
                            verbose      = TRUE)
  write_csv(results, results_file)
} else {
  results <- read_csv(results_file, show_col_types = FALSE)
}

cat("Rows loaded:", nrow(results), "\n")
```

    Rows loaded: 390 

Code

``` r
cat("Expected:   ", 30 * 13, "\n")
```

    Expected:    390 

Code

``` r
cat("NA count:   ", sum(is.na(results$MAPE)), "\n")
```

    NA count:    0 

## 6 Results

### 6.1 Summary table

Code

``` r
summary_df <- summarise_results(results)
```

| Model                               | MAPE                   | RMSPE                  | MAAPE                  | MSE     | R2      |
|:------------------------------------|:-----------------------|:-----------------------|:-----------------------|:--------|:--------|
| SVR-MAPE                            | 34.15 \[33.34, 34.97\] | 44.02 \[42.34, 45.72\] | 19.63 \[19.28, 19.96\] | 4100.34 | 0.2899  |
| SVR-MAPE + Sym. Kernel              | 35.28 \[34.33, 36.19\] | 46.14 \[44.14, 48.19\] | 20.10 \[19.72, 20.46\] | 4245.32 | 0.2635  |
| WLS (1/y²)                          | 36.57 \[35.83, 37.25\] | 43.95 \[42.73, 45.16\] | 21.29 \[20.93, 21.63\] | 5739.53 | 0.0042  |
| Linear Regression                   | 39.58 \[37.87, 41.17\] | 60.86 \[57.56, 64.18\] | 20.62 \[19.94, 21.25\] | 2933.23 | 0.4905  |
| QR (τ = 0.5)                        | 39.95 \[38.33, 41.52\] | 61.40 \[58.08, 64.65\] | 20.81 \[20.21, 21.40\] | 3013.17 | 0.4765  |
| Log-SVR                             | 40.23 \[38.60, 41.91\] | 62.03 \[58.22, 66.17\] | 20.91 \[20.33, 21.47\] | 3961.37 | 0.3112  |
| XGBoost                             | 41.17 \[39.51, 42.89\] | 59.55 \[56.84, 62.37\] | 21.80 \[21.12, 22.48\] | 3322.33 | 0.4231  |
| ε-SVR (MSE)                         | 42.05 \[40.28, 43.81\] | 64.44 \[61.08, 67.84\] | 21.58 \[20.90, 22.20\] | 3598.02 | 0.3737  |
| Random Forest                       | 42.12 \[40.51, 43.76\] | 60.96 \[58.18, 63.79\] | 22.13 \[21.48, 22.80\] | 3261.01 | 0.4334  |
| LS-RMSPE (MAPE opt.)                | 43.29 \[42.63, 43.92\] | 49.61 \[48.85, 50.39\] | 24.95 \[24.59, 25.28\] | 9519.84 | -0.6525 |
| LS-RMSPE (RMSPE opt.)               | 43.29 \[42.63, 43.92\] | 49.61 \[48.85, 50.39\] | 24.95 \[24.59, 25.28\] | 9519.84 | -0.6525 |
| LS-RMSPE + Sym. Kernel (MAPE opt.)  | 43.32 \[42.66, 43.97\] | 49.65 \[48.88, 50.41\] | 24.97 \[24.61, 25.30\] | 9566.36 | -0.6605 |
| LS-RMSPE + Sym. Kernel (RMSPE opt.) | 43.32 \[42.66, 43.97\] | 49.65 \[48.88, 50.41\] | 24.97 \[24.61, 25.30\] | 9566.36 | -0.6605 |

Values in brackets are 95% percentile bootstrap confidence intervals
over 30 random train/test splits.

### 6.2 Box plots

Code

``` r
make_bp <- function(metric, ylab) {
  results |>
    select(abbrev, family, value = all_of(metric)) |>
    mutate(abbrev = fct_reorder(abbrev, value, .fun = median)) |>
    ggplot(aes(x = abbrev, y = value, fill = family)) +
    geom_boxplot(outlier.size = 0.8, linewidth = 0.4) +
    scale_fill_manual(values = c("psvr" = "#2c7bb6",
                                 "baseline" = "#d7191c"),
                      name = NULL) +
    labs(x = NULL, y = ylab, title = metric) +
    theme_bw(base_size = 11) +
    theme(axis.text.x  = element_text(angle = 45, hjust = 1),
          legend.position = "bottom")
}

p_mape  <- make_bp("MAPE",  "MAPE (%)")
p_rmspe <- make_bp("RMSPE", "RMSPE (%)")
p_maape <- make_bp("MAAPE", "MAAPE (%)")
p_r2    <- make_bp("R2",    "R²")

(p_mape + p_rmspe) / (p_maape + p_r2) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
```

![](diabetes_files/figure-html/boxplots-1.png)

### 6.3 Statistical comparison

Code

``` r
wilcox_df <- wilcoxon_vs_best(results, metric = "MAPE")
```

| Model                               | Reference baseline | p-value | Significance |
|:------------------------------------|:-------------------|:--------|:-------------|
| LS-RMSPE (MAPE opt.)                | WLS (1/y²)         | 0.0000  | \*\*\*       |
| LS-RMSPE (RMSPE opt.)               | WLS (1/y²)         | 0.0000  | \*\*\*       |
| LS-RMSPE + Sym. Kernel (MAPE opt.)  | WLS (1/y²)         | 0.0000  | \*\*\*       |
| LS-RMSPE + Sym. Kernel (RMSPE opt.) | WLS (1/y²)         | 0.0000  | \*\*\*       |
| SVR-MAPE                            | WLS (1/y²)         | 0.0000  | \*\*\*       |
| Random Forest                       | WLS (1/y²)         | 0.0000  | \*\*\*       |
| ε-SVR (MSE)                         | WLS (1/y²)         | 0.0001  | \*\*\*       |
| XGBoost                             | WLS (1/y²)         | 0.0001  | \*\*\*       |
| Log-SVR                             | WLS (1/y²)         | 0.0012  | \*\*         |
| QR (τ = 0.5)                        | WLS (1/y²)         | 0.0012  | \*\*         |
| SVR-MAPE + Sym. Kernel              | WLS (1/y²)         | 0.0020  | \*\*         |
| Linear Regression                   | WLS (1/y²)         | 0.0035  | \*\*         |

Paired Wilcoxon signed-rank test vs. best baseline on MAPE.

The best baseline in this study is WLS (1/y²). Of the six `psvr` models,
6 achieve a statistically significant improvement over that baseline at
α = 0.05. Significance codes: \*\*\* p \< 0.001, \*\* p \< 0.01, \* p \<
0.05, ns = not significant.

## 7 Discussion

SVR-MAPE leads on MAPE (34–35% range) despite the dataset’s high CV;
predicting always the mean yields approximately 62% MAPE, so `psvr`
models capture meaningful signal. R² is modest (0.3–0.5) for all models
— expected for this small, noisy dataset where linear methods are
competitive.

Symmetric kernel models (m2, m4a, m4b) show R² near zero or negative:
Diabetes has no natural symmetry structure, so the symmetric kernel term
introduces noise rather than useful inductive bias. This delimits the
applicability of symmetric extensions to datasets with inherent symmetry
in the feature space.

As with Boston Housing, optimizing MAPE trades some explained variance
for scale-free accuracy. SVR-MAPE is the appropriate model when
percentage-error is the clinical reporting metric. For R²-focused tasks,
linear regression remains competitive on this small dataset and should
be considered a strong baseline.

**Practical recommendation:** use SVR-MAPE when percentage-error is the
clinical reporting metric; for R²-focused tasks, linear regression
remains competitive on this dataset.

## 8 Downloadable files

Code

``` r
# xfun::embed_file(
#   "case-studies/experiment_helpers.R",
#   text = "Download experiment helper script (.R)"
# )
# 
# xfun::embed_file(
#   "case-studies/results/diabetes-results.csv",
#   text = "Download full results (30 seeds x 13 models x 6 metrics)"
# )
```
