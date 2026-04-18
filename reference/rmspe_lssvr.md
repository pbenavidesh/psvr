# Fit LS-SVR with RMSPE loss (Model 3)

Solves the linear system derived in Theorem 3 of Benavides-Herrera et
al. (2026). The primal objective is `½‖ω‖² + (Γ/2) Σ eₖ²/yₖ²`, leading
to the (N+1)×(N+1) system

## Usage

``` r
rmspe_lssvr(X, y, kernel, gamma)
```

## Arguments

- X:

  Numeric matrix of training inputs, one observation per row (N × p).

- y:

  Numeric vector of training targets, length N. Must satisfy `y > 0`.

- kernel:

  A kernel function created by
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- gamma:

  Regularization parameter `Γ > 0`.

## Value

An object of class `"psvr_rmspe"`, a list with components:

- `alpha`:

  Numeric vector of dual variables (length N).

- `b`:

  Numeric scalar bias term.

- `X_train`:

  The training matrix `X` (kept for prediction).

- `kernel`:

  The kernel function used.

## Details

    [ 0   1ᵀ     ] [ b ]   [ 0 ]
    [ 1   Ω + YΓ ] [ α ] = [ y ]

where `YΓ = diag(y₁²/Γ, …, yN²/Γ)` is added to the diagonal of Ω.

## Examples

``` r
X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- rmspe_lssvr(X, y, kernel = K, gamma = 1)
predict(fit, X)
#> [1] 2.743967 2.904833 2.969809
```
