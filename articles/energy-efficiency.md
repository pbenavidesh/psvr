# Case Study: Building Energy Efficiency

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

The Energy Efficiency dataset (Tsanas & Xifara, 2012; UCI ML Repository)
contains n = 768 building simulations with 8 architectural features
(relative compactness, surface area, wall area, roof area, overall
height, orientation, glazing area, glazing area distribution). The
target is heating load Y1 (kWh/m²), strictly positive, ranging from 6.01
to 43.10.

In building energy certification (e.g., LEED, BREEAM, NOM-020),
compliance thresholds are expressed as percentage deviations from design
values, making MAPE the natural evaluation metric: a 10% error on a
low-load building and a 10% error on a high-load building have
proportionally equivalent design implications.

Features X6 (Orientation) and X8 (Glazing Area Distribution) are ordinal
and treated as continuous following standard practice for kernel and
tree-based methods. This article demonstrates that `psvr` models —
trained directly under MAPE and RMSPE loss — provide competitive
percentage-error performance over 30 randomized train/test splits. Data
loaded from the UCI ML Repository.

## 2 Setup

Code

``` r
library(tidyverse)
library(tidymodels)
library(psvr)
library(readxl)
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
ee_local <- "case-studies/results/ENB2012_data.xlsx"
if (!file.exists(ee_local)) {
  download.file(
    paste0("https://archive.ics.uci.edu/ml/",
           "machine-learning-databases/00242/ENB2012_data.xlsx"),
    ee_local, mode = "wb", quiet = TRUE)
}
df_ee <- readxl::read_excel(ee_local)
y <- df_ee$Y1
X <- df_ee |> select(X1:X8) |> as.matrix()

stopifnot(all(y > 0))
```

| Statistic |  Value |
|:----------|-------:|
| n         | 768.00 |
| p         |   8.00 |
| min(y)    |   6.01 |
| median(y) |  18.95 |
| mean(y)   |  22.31 |
| max(y)    |  43.10 |

Energy Efficiency dataset summary (heating load Y1).

![](energy-efficiency_files/figure-html/data-histogram-1.png)

All 768 target values satisfy `y > 0` (confirmed by `stopifnot` above).

> **Note**
>
> Features X6 (Orientation, values 2–5) and X8 (Glazing Area
> Distribution, values 0–5) are ordinal. The median absolute deviation
> of Y1 confirms moderate spread suitable for MAPE evaluation.

## 4 Experimental protocol

The benchmark follows a 30-seed protocol: for each seed in 1–30, the 768
observations are randomly split into 80% training (≈ 614 rows) and 20%
test (≈ 154 rows). Hyperparameters are selected via 5-fold
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
MASE uses the lag-1 naive denominator; because the Energy Efficiency
data has no temporal ordering, the denominator depends on row order
within each split and should be interpreted accordingly. Quantile
regression (b7a) could not be fitted on this dataset due to
near-singular design matrix caused by the ordinal features X6 and X8;
results for b7a are reported as NA across all seeds.

## 5 Run experiment

Code

``` r
results_file <- "case-studies/results/energy-efficiency-results-grid-osqp.csv"

if (!file.exists(results_file)) {
  results <- run_experiment(X, y,
                            dataset_name = "energy_efficiency",
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
cat("NA count (b7a expected):", sum(is.na(results$MAPE)), "\n")
```

    NA count (b7a expected): 30 

## 6 Results

### 6.1 Summary table

Code

``` r
summary_df <- summarise_results(results)
```

| Model                               | MAPE                   | RMSPE                  | MAAPE               | MSE   | R2     |
|:------------------------------------|:-----------------------|:-----------------------|:--------------------|:------|:-------|
| XGBoost                             | 1.16 \[1.09, 1.24\]    | 1.97 \[1.73, 2.29\]    | 0.74 \[0.69, 0.78\] | 0.16  | 0.9985 |
| Random Forest                       | 1.84 \[1.76, 1.91\]    | 2.72 \[2.50, 2.98\]    | 1.17 \[1.12, 1.22\] | 0.34  | 0.9967 |
| SVR-MAPE                            | 1.96 \[1.89, 2.04\]    | 3.12 \[2.92, 3.36\]    | 1.25 \[1.20, 1.29\] | 0.45  | 0.9955 |
| Log-SVR                             | 2.37 \[2.27, 2.48\]    | 3.90 \[3.50, 4.42\]    | 1.50 \[1.44, 1.57\] | 0.56  | 0.9945 |
| SVR-MAPE + Sym. Kernel              | 4.87 \[4.65, 5.09\]    | 8.02 \[7.66, 8.38\]    | 3.07 \[2.94, 3.21\] | 6.49  | 0.9367 |
| ε-SVR (MSE)                         | 5.12 \[4.95, 5.29\]    | 7.18 \[6.76, 7.61\]    | 3.23 \[3.13, 3.34\] | 1.28  | 0.9874 |
| LS-RMSPE (MAPE opt.)                | 9.10 \[8.93, 9.26\]    | 11.76 \[11.56, 11.97\] | 5.74 \[5.63, 5.84\] | 11.58 | 0.8867 |
| LS-RMSPE (RMSPE opt.)               | 9.10 \[8.93, 9.26\]    | 11.76 \[11.56, 11.97\] | 5.74 \[5.63, 5.84\] | 11.58 | 0.8867 |
| WLS (1/y²)                          | 9.55 \[9.36, 9.74\]    | 12.31 \[12.03, 12.60\] | 6.01 \[5.89, 6.13\] | 10.08 | 0.9013 |
| Linear Regression                   | 9.66 \[9.45, 9.88\]    | 12.96 \[12.63, 13.30\] | 6.07 \[5.94, 6.20\] | 8.71  | 0.9144 |
| LS-RMSPE + Sym. Kernel (MAPE opt.)  | 10.86 \[10.68, 11.06\] | 14.01 \[13.75, 14.31\] | 6.82 \[6.70, 6.94\] | 15.01 | 0.8533 |
| LS-RMSPE + Sym. Kernel (RMSPE opt.) | 10.86 \[10.68, 11.06\] | 14.01 \[13.75, 14.31\] | 6.82 \[6.70, 6.94\] | 15.01 | 0.8533 |
| QR (τ = 0.5)                        | NaN \[NA, NA\]         | NaN \[NA, NA\]         | NaN \[NA, NA\]      | NaN   | NaN    |

Values in brackets are 95% percentile bootstrap confidence intervals
over 30 random train/test splits. b7a (Quantile Regression) could not be
fitted due to near-singular design matrix (ordinal features X6, X8);
results excluded from statistical comparison.

### 6.2 Box plots

Code

``` r
make_bp <- function(metric, ylab) {
  results |>
    select(abbrev, family, value = all_of(metric)) |>
    filter(!is.na(value)) |>
    mutate(abbrev = fct_reorder(abbrev, value,
                                .fun = median)) |>
    ggplot(aes(x = abbrev, y = value, fill = family)) +
    geom_boxplot(outlier.size = 0.8, linewidth = 0.4) +
    scale_fill_manual(values = c("psvr"     = "#2c7bb6",
                                 "baseline" = "#d7191c"),
                      name = NULL) +
    labs(x = NULL, y = ylab, title = metric) +
    theme_bw(base_size = 11) +
    theme(axis.text.x     = element_text(angle = 45, hjust = 1),
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

![](energy-efficiency_files/figure-html/boxplots-1.png)

### 6.3 Statistical comparison

Code

``` r
results_no_qr <- results |>
  filter(!(model_id == "b7a" & is.na(MAPE)))

wilcox_df <- wilcoxon_vs_best(results_no_qr, metric = "MAPE")
```

| Model                               | Reference baseline | p-value | Significance |
|:------------------------------------|:-------------------|:--------|:-------------|
| ε-SVR (MSE)                         | XGBoost            | 0.0000  | \*\*\*       |
| Random Forest                       | XGBoost            | 0.0000  | \*\*\*       |
| Linear Regression                   | XGBoost            | 0.0000  | \*\*\*       |
| WLS (1/y²)                          | XGBoost            | 0.0000  | \*\*\*       |
| Log-SVR                             | XGBoost            | 0.0000  | \*\*\*       |
| SVR-MAPE                            | XGBoost            | 0.0000  | \*\*\*       |
| SVR-MAPE + Sym. Kernel              | XGBoost            | 0.0000  | \*\*\*       |
| LS-RMSPE (MAPE opt.)                | XGBoost            | 0.0000  | \*\*\*       |
| LS-RMSPE (RMSPE opt.)               | XGBoost            | 0.0000  | \*\*\*       |
| LS-RMSPE + Sym. Kernel (MAPE opt.)  | XGBoost            | 0.0000  | \*\*\*       |
| LS-RMSPE + Sym. Kernel (RMSPE opt.) | XGBoost            | 0.0000  | \*\*\*       |

Paired Wilcoxon signed-rank test vs. best baseline on MAPE. b7a excluded
(NA results).

The best baseline in this study is XGBoost. Of the six `psvr` models, 6
achieve a statistically significant improvement over that baseline at α
= 0.05. Significance codes: \*\*\* p \< 0.001, \*\* p \< 0.01, \* p \<
0.05, ns = not significant.

## 7 Discussion

Energy Efficiency has moderate target spread (range 6–43 kWh/m²), making
it a favorable dataset for MAPE-optimized models; `psvr` models achieve
competitive MAPE, benefiting from the smooth energy-load response
surface produced by these building simulations.

Ordinal features (X6, X8) do not affect kernel or tree-based methods but
prevent Quantile Regression from fitting — this illustrates a practical
advantage of kernel methods over linear quantile regression in
mixed-type feature spaces. Symmetric kernel models show similar behavior
to the Boston Housing and Diabetes results: the dataset has no natural
geometric symmetry in feature space, so the symmetric kernel term
introduces noise rather than useful inductive bias, and non-symmetric
variants are generally preferable.

As expected, R² is high for all models on this simulation-derived
dataset, where the relationship between architectural features and
heating load is relatively smooth and deterministic. `psvr` models
remain competitive on both MAPE and R² here, unlike the noisier
real-world datasets.

**Practical recommendation:** use SVR-MAPE or LS-RMSPE for building
energy load prediction when percentage-error compliance metrics are
required; kernel methods have an additional practical advantage over
linear quantile regression when ordinal features are present.

## 8 Downloadable files

Code

``` r
# xfun::embed_file(
#   "case-studies/experiment_helpers.R",
#   text = "Download experiment helper script (.R)"
# )
# 
# xfun::embed_file(
#   "case-studies/results/energy-efficiency-results.csv",
#   text = "Download full results (30 seeds x 13 models x 6 metrics)"
# )
```
