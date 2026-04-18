# Fit symmetric LS-SVR with RMSPE loss (Model 4)

Solves the linear system derived in Theorem 4 of Benavides-Herrera et
al. (2026). Identical in structure to
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
(Model 3) but replaces the kernel matrix Ω with the symmetrized matrix
`Ωs = ½(Ω + a·Ω*)`, where `Ω*ₖₗ = K(xₖ, -xₗ)`. The system solved is

## Usage

``` r
rmspe_sym_lssvr(X, y, kernel, gamma, a = 1)
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

- a:

  Symmetry parameter: `1` for even symmetry `f(x) = f(-x)`, `-1` for odd
  symmetry `f(x) = -f(-x)`.

## Value

An object of class `"psvr_rmspe_sym"`, a list with components:

- `alpha`:

  Numeric vector of dual variables (length N).

- `b`:

  Numeric scalar bias term.

- `X_train`:

  The training matrix `X` (kept for prediction).

- `kernel`:

  The kernel function used.

- `gamma`:

  The regularization parameter `Γ`.

- `a`:

  The symmetry parameter.

- `n_train`:

  Number of training observations.

- `p_train`:

  Number of training features (columns).

## Details

    [ 0   1ᵀ      ] [ b ]   [ 0 ]
    [ 1   Ωs + YΓ ] [ α ] = [ y ]

where `YΓ = diag(y₁²/Γ, …, yN²/Γ)`.

The kernel must satisfy Assumption 3 of the paper (kernel symmetry):
`K(-xi, xj) = K(xi, -xj)` and `K(-xi, -xj) = K(xi, xj)`. RBF and
even-degree polynomial kernels satisfy this; see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Examples

``` r
X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- rmspe_sym_lssvr(X, y, kernel = K, gamma = 1, a = 1)
predict(fit, X)
#> [1] 2.769355 2.853120 2.886170
```
