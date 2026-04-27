# Fit epsilon-SVR with MAPE loss (Model 1)

Solves the quadratic program derived in Theorem 1 of Benavides-Herrera
et al. (2026) via `osqp`. The dual is expressed in variables
`u = [α; α*] ∈ R^{2N}` with:

## Usage

``` r
mape_svr(X, y, kernel, C, eps, solver = c("smo", "osqp"), tol = 1e-05)
```

## Arguments

- X:

  Numeric matrix of training inputs, one observation per row (N × p).

- y:

  Numeric vector of training targets, length N. Must satisfy `y > 0`.

- kernel:

  A kernel function created by
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

- C:

  Regularization parameter `C > 0`.

- eps:

  Insensitivity tube half-width `ε ≥ 0` (in percentage units, i.e., the
  tube is `ε%` of each target).

- solver:

  Backend that solves the dual QP. `"smo"` (default) uses an internal
  libsvm-style SMO solver with no third-party dependency; `"osqp"`
  delegates to the `osqp` package (which then becomes a soft
  requirement). Both backends solve the same QP and return the same
  support vectors and bias up to numerical tolerance.

- tol:

  Threshold below which `|βk|` is treated as zero when selecting support
  vectors and free support vectors (default `1e-5`).

## Value

An object of class `"psvr_mape"`, a list with components:

- `beta`:

  Numeric vector of non-zero dual differences `βk` (length equal to the
  number of support vectors).

- `b`:

  Numeric scalar bias term.

- `X_sv`:

  Numeric matrix of support vector inputs.

- `y_sv`:

  Numeric vector of support vector targets.

- `kernel`:

  The kernel function used.

- `C`:

  The regularization parameter `C`.

- `eps`:

  The `ε` value used.

- `n_train`:

  Number of training observations.

- `p_train`:

  Number of training features (columns).

## Details

- **P** = `[Ω, -Ω; -Ω, Ω]` (2N × 2N, upper triangular passed to osqp)

- **q** = `[y(ε/100 − 1); y(1 + ε/100)]`

- **Equality:** `[1ᵀ, −1ᵀ] u = 0`

- **Box:** `0 ≤ αk ≤ 100C/yk`, `0 ≤ αk* ≤ 100C/yk`

After solving, `βk = αk − αk*` and only support vectors with
`|βk| > tol` are retained.

## Examples

``` r
X <- matrix(1:6, ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- mape_svr(X, y, kernel = K, C = 10, eps = 5)
predict(fit, X)
#> [1] 2.207585 3.988709 5.888707
```
