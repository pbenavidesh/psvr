# psvr — Percentage-Error Support Vector Regression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19643526.svg)](https://doi.org/10.5281/zenodo.19643526)

`psvr` implements four support vector regression models derived from a
unified mathematical framework for percentage-error loss functions.
Classical SVR minimises absolute-error losses (MAE, MSE), which are
scale-dependent: an error of 1 unit is negligible when the target is 1
000 but critical when it is 2. `psvr` addresses this by optimising MAPE
and RMSPE directly, making it well-suited for forecasting tasks where
targets are strictly positive and relative accuracy is what matters.

| Function                                                                               | Loss                     | Solver                   |
|----------------------------------------------------------------------------------------|--------------------------|--------------------------|
| [`mape_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_svr.md)               | MAPE — ε-insensitive     | quadratic program (osqp) |
| [`mape_sym_svr()`](https://pbenavidesh.github.io/psvr/reference/mape_sym_svr.md)       | MAPE — symmetric kernel  | quadratic program (osqp) |
| [`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)         | RMSPE — least-squares    | linear system            |
| [`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md) | RMSPE — symmetric kernel | linear system            |

All models require **strictly positive targets** (`y > 0`).

## Installation

``` r
# Development version from GitHub
remotes::install_github("pbenavidesh/psvr")
```

## Quick start

``` r
library(psvr)

# Synthetic dataset: even function (f(-x) = f(x)), targets strictly positive
set.seed(42)
n  <- 100
X  <- matrix(rnorm(n * 2), n, 2)
y  <- 2 + X[, 1]^2 + 0.5 * X[, 2]^2 + rnorm(n, sd = 0.1)

tr <- 1:70;  te <- 71:100
X_tr <- X[tr, ];  y_tr <- y[tr]
X_te <- X[te, ];  y_te <- y[te]

# Standardise features using training-set statistics (important for RBF)
col_mean <- colMeans(X_tr);  col_sd <- apply(X_tr, 2, sd)
X_tr_s   <- scale(X_tr, col_mean, col_sd)
X_te_s   <- scale(X_te, col_mean, col_sd)

K <- make_kernel("rbf", sigma = 1)

# Model 1 — ε-SVR with MAPE
fit1  <- mape_svr(X_tr_s, y_tr, kernel = K, C = 0.5, eps = 5)
pred1 <- predict(fit1, X_te_s)

# Model 2 — Symmetric ε-SVR with MAPE  (a = 1: even-function prior)
fit2  <- mape_sym_svr(X_tr_s, y_tr, kernel = K, C = 0.5, eps = 5, a = 1)
pred2 <- predict(fit2, X_te_s)

# Model 3 — LS-SVR with RMSPE
fit3  <- rmspe_lssvr(X_tr_s, y_tr, kernel = K, gamma = 100)
pred3 <- predict(fit3, X_te_s)

# Model 4 — Symmetric LS-SVR with RMSPE  (a = 1: even-function prior)
fit4  <- rmspe_sym_lssvr(X_tr_s, y_tr, kernel = K, gamma = 100, a = 1)
pred4 <- predict(fit4, X_te_s)

mape <- function(y, yhat) mean(abs(y - yhat) / y) * 100

cat(sprintf("Model 1 (MAPE e-SVR):       MAPE = %.2f%%\n", mape(y_te, pred1)))
cat(sprintf("Model 2 (MAPE sym e-SVR):   MAPE = %.2f%%\n", mape(y_te, pred2)))
cat(sprintf("Model 3 (RMSPE LS-SVR):     MAPE = %.2f%%\n", mape(y_te, pred3)))
cat(sprintf("Model 4 (RMSPE sym LS-SVR): MAPE = %.2f%%\n", mape(y_te, pred4)))
#> Model 1 (MAPE e-SVR):       MAPE = 3.59%
#> Model 2 (MAPE sym e-SVR):   MAPE = 6.04%
#> Model 3 (RMSPE LS-SVR):     MAPE = 3.76%
#> Model 4 (RMSPE sym LS-SVR): MAPE = 5.79%
```

For a full worked example on the Boston Housing dataset — including a
70/30 train–test split, feature standardisation, and comparison against
a linear baseline — see
[`vignette("getting-started", package = "psvr")`](https://pbenavidesh.github.io/psvr/articles/getting-started.md).

## Reference

Benavides-Herrera, P., Álvarez-Álvarez, G., Ruiz-Cruz, R., &
Sánchez-Torres, J. D. (2026). A unified family of percentage-error
support vector regression models with symmetric kernel extensions.
*Mathematics*, MDPI. <https://doi.org/10.3390/math>
