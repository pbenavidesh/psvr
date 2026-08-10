
<!-- README.md is generated from README.Rmd. Please edit that file -->

# psvr <img src="man/figures/logo.png" align="right" height="139"/>

<!-- badges: start -->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19643526.svg)](https://doi.org/10.5281/zenodo.19643526)
<!-- badges: end -->

`psvr` implements four support vector regression models derived from a
unified mathematical framework for percentage-error loss functions.
Classical SVR minimises absolute-error losses (MAE, MSE), which are
scale-dependent: an error of 1 unit is negligible when the target is 1
000 but critical when it is 2. `psvr` addresses this by optimising MAPE
and RMSPE directly, making it well-suited for forecasting tasks where
targets are strictly positive and relative accuracy is what matters.

`psvr()` is the single entry point; `loss` and `sym` select the model:

| `loss`    | `sym`         | Model                       | Solver                  |
|-----------|---------------|-----------------------------|-------------------------|
| `"mape"`  | `NULL`        | ε-SVR with MAPE             | SMO (default) or `osqp` |
| `"mape"`  | `+1L` / `-1L` | Symmetric ε-SVR with MAPE   | SMO (default) or `osqp` |
| `"rmspe"` | `NULL`        | LS-SVR with RMSPE           | linear system           |
| `"rmspe"` | `+1L` / `-1L` | Symmetric LS-SVR with RMSPE | linear system           |

`sym = +1L` imposes an even-function prior (`f(-x) = f(x)`), `sym = -1L`
an odd one. All models require **strictly positive targets** (`y > 0`).

## Installation

``` r
# CRAN
install.packages("psvr")

# Development version from GitHub
pak::pak("pbenavidesh/psvr")
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
fit1  <- psvr(X_tr_s, y_tr, loss = "mape", kernel = K, C = 0.5, eps = 5)
pred1 <- predict(fit1, X_te_s)

# Model 2 — Symmetric ε-SVR with MAPE  (sym = 1L: even-function prior)
fit2  <- psvr(X_tr_s, y_tr, loss = "mape", sym = 1L, kernel = K, C = 0.5, eps = 5)
pred2 <- predict(fit2, X_te_s)

# Model 3 — LS-SVR with RMSPE
fit3  <- psvr(X_tr_s, y_tr, loss = "rmspe", kernel = K, gamma = 100)
pred3 <- predict(fit3, X_te_s)

# Model 4 — Symmetric LS-SVR with RMSPE  (sym = 1L: even-function prior)
fit4  <- psvr(X_tr_s, y_tr, loss = "rmspe", sym = 1L, kernel = K, gamma = 100)
pred4 <- predict(fit4, X_te_s)

mape <- function(y, yhat) mean(abs(y - yhat) / y) * 100

cat(sprintf("Model 1 (MAPE e-SVR):       MAPE = %.2f%%", mape(y_te, pred1)),
    sprintf("Model 2 (MAPE sym e-SVR):   MAPE = %.2f%%", mape(y_te, pred2)),
    sprintf("Model 3 (RMSPE LS-SVR):     MAPE = %.2f%%", mape(y_te, pred3)),
    sprintf("Model 4 (RMSPE sym LS-SVR): MAPE = %.2f%%", mape(y_te, pred4)),
    sep = "\n")
#> Model 1 (MAPE e-SVR):       MAPE = 3.57%
#> Model 2 (MAPE sym e-SVR):   MAPE = 5.97%
#> Model 3 (RMSPE LS-SVR):     MAPE = 3.76%
#> Model 4 (RMSPE sym LS-SVR): MAPE = 5.79%
```

For a full worked example on the Boston Housing dataset — including a
70/30 train–test split, feature standardisation, and comparison against
a linear baseline — see `vignette("getting-started", package = "psvr")`.

## Reference

Benavides-Herrera, P., Álvarez, G., Ruiz-Cruz, R., & Sánchez-Torres, J.
D. (2026). A unified family of percentage-error support vector
regression models with symmetric kernel extensions. *Mathematics*,
14(10), 1679. <https://doi.org/10.3390/math14101679>
