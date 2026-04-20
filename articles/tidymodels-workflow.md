# tidymodels workflow with psvr

This vignette demonstrates the full tidymodels pipeline with **psvr**:
data splitting, preprocessing, hyperparameter tuning by
cross-validation, and final model evaluation. We use
[`psvr_rmspe_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe_specs.md)
(LS-SVR with RMSPE loss, RBF kernel) and tune the regularisation
parameter `cost` ($\Gamma$) against MAPE.

``` r
library(psvr)
library(parsnip)
library(rsample)
library(recipes)
library(workflows)
library(tune)
library(dials)
library(yardstick)
```

## Data

The synthetic even-function dataset from the package README:
$y = 2 + x_{1}^{2} + 0.5\, x_{2}^{2} + \varepsilon$,
$\varepsilon \sim \mathcal{N}\left( 0,\, 0.1^{2} \right)$. Targets are
strictly positive by construction ($y > 0$).

``` r
set.seed(42)
n   <- 200
x1  <- runif(n, -3, 3)
x2  <- runif(n, -3, 3)
y   <- 2 + x1^2 + 0.5 * x2^2 + rnorm(n, sd = 0.1)
dat <- data.frame(y = y, x1 = x1, x2 = x2)
```

## 1 — Split

``` r
set.seed(1)
split <- initial_split(dat, prop = 0.75)
train <- training(split)
test  <- testing(split)
```

## 2 — Preprocessing recipe

Centre and scale all predictors so the RBF kernel operates on a
standardised feature space.

``` r
rec <- recipe(y ~ x1 + x2, data = train) |>
  step_normalize(all_predictors())
```

## 3 — Model spec with `tune()`

Both `cost` (maps to $\Gamma$) and `rbf_sigma` (the RBF bandwidth
$\sigma$) are
[`tune()`](https://hardhat.tidymodels.org/reference/tune.html)
placeholders; the grid search will explore all combinations.

``` r
spec <- psvr_rmspe_rbf(cost = tune(), rbf_sigma = tune()) |>
  set_engine("psvr")
```

## 4 — Workflow

``` r
wf <- workflow() |>
  add_recipe(rec) |>
  add_model(spec)
```

## 5 — Tune with 5-fold CV

We search over a $5 \times 3$ grid of `cost` and `rbf_sigma` values (15
combinations) and evaluate each fold by MAPE.
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
only solves a linear system, so 75 fits complete in seconds.

``` r
set.seed(2)
folds <- vfold_cv(train, v = 5)

cost_grid <- tidyr::crossing(
  cost      = c(0.01, 0.1, 1, 10, 100),
  rbf_sigma = c(0.5, 1, 2)
)

tune_res <- tune_grid(
  wf,
  resamples = folds,
  grid      = cost_grid,
  metrics   = metric_set(mape)
)
```

Cross-validated MAPE for each candidate (lower is better):

``` r
collect_metrics(tune_res)[, c("cost", "rbf_sigma", "mean", "std_err")]
#> # A tibble: 15 × 4
#>      cost rbf_sigma  mean std_err
#>     <dbl>     <dbl> <dbl>   <dbl>
#>  1   0.01       0.5 41.8   1.07  
#>  2   0.01       1   41.8   1.07  
#>  3   0.01       2   42.0   1.06  
#>  4   0.1        0.5 40.0   1.09  
#>  5   0.1        1   40.3   1.08  
#>  6   0.1        2   41.6   1.07  
#>  7   1          0.5 29.9   1.02  
#>  8   1          1   30.2   1.10  
#>  9   1          2   38.6   1.11  
#> 10  10          0.5 13.6   0.366 
#> 11  10          1   12.4   0.398 
#> 12  10          2   23.1   1.02  
#> 13 100          0.5  4.68  0.104 
#> 14 100          1    4.88  0.0736
#> 15 100          2    6.30  0.203
```

## 6 — Select best

``` r
best_params <- select_best(tune_res, metric = "mape")
best_params
#> # A tibble: 1 × 3
#>    cost rbf_sigma .config         
#>   <dbl>     <dbl> <chr>           
#> 1   100       0.5 pre0_mod13_post0
```

## 7 — Final fit and test-set evaluation

[`last_fit()`](https://tune.tidymodels.org/reference/last_fit.html)
refits on the full training set with the chosen `cost` and evaluates
once on the held-out test data.

``` r
final_wf  <- finalize_workflow(wf, best_params)
final_fit <- last_fit(final_wf, split, metrics = metric_set(mape))

collect_metrics(final_fit)
#> # A tibble: 1 × 4
#>   .metric .estimator .estimate .config        
#>   <chr>   <chr>          <dbl> <chr>          
#> 1 mape    standard        5.24 pre0_mod0_post0
```

Predictions on the test set:

``` r
preds <- collect_predictions(final_fit)
head(preds[, c(".row", "y", ".pred")])
#> # A tibble: 6 × 3
#>    .row     y .pred
#>   <int> <dbl> <dbl>
#> 1     3  5.99  6.00
#> 2     4  6.20  6.25
#> 3     5  4.69  4.90
#> 4     6  1.96  2.14
#> 5     8  6.70  6.87
#> 6     9  3.93  3.96
```

The fitted workflow can also be used directly for new data:

``` r
new_obs <- data.frame(x1 = c(0, 1, -2), x2 = c(0, 1, 2))
predict(extract_workflow(final_fit), new_data = new_obs)
#> # A tibble: 3 × 1
#>   .pred
#>   <dbl>
#> 1  2.06
#> 2  3.51
#> 3  8.15
```

## 8 — Inspecting the fitted psvr model

The tidymodels layer wraps a `psvr_rmspe` object (returned by
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)).
Extract it to use [`print()`](https://rdrr.io/r/base/print.html) and
[`coef()`](https://rdrr.io/r/stats/coef.html) directly.

``` r
# extract_fit_engine() unwraps the parsnip/workflow layer to the raw psvr object
psvr_fit <- extract_fit_engine(extract_workflow(final_fit))
print(psvr_fit)
#> 
#> LS-SVR with RMSPE loss  [psvr_rmspe]
#> 
#>   Kernel:        RBF (sigma = 0.5)
#>   Gamma:         100
#>   Training obs.: 150
```

``` r
cf <- coef(psvr_fit)
# alpha: N dual variables; weight each training point in f(x) = sum_k alpha_k K(x_k, x) + b
# b:     bias / intercept term
# X_sv:  all N training inputs (LS-SVR has no sparsity — every training point contributes)
cat(sprintf("b = %.4f  |  alpha range: [%.4f, %.4f]\n",
            cf$b, min(cf$alpha), max(cf$alpha)))
#> b = 7.6271  |  alpha range: [-3.5482, 2.1956]
```
