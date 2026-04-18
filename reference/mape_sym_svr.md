# Fit symmetric epsilon-SVR with MAPE loss (Model 2)

Solves the quadratic program derived in Theorem 2 of Benavides-Herrera
et al. (2026) via `osqp`. The symmetry constraint `f(x) = a·f(-x)` is
enforced by replacing the kernel with
`Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)`.

## Usage

``` r
mape_sym_svr(X, y, kernel, C, eps, a = 1, tol = 1e-05)
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

  Insensitivity tube half-width `ε ≥ 0` (in percentage units).

- a:

  Symmetry parameter: `1` for even symmetry `f(x) = f(-x)`, `-1` for odd
  symmetry `f(x) = -f(-x)`.

- tol:

  Threshold below which `|βk|` is treated as zero (default `1e-5`).

## Value

An object of class `"psvr_mape_sym"`, a list with components:

- `beta`:

  Numeric vector of non-zero dual differences `βk`.

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

- `a`:

  The symmetry parameter.

- `n_train`:

  Number of training observations.

- `p_train`:

  Number of training features (columns).

## Details

The dual in variables `u = [α; α*] ∈ R^{2N}` is:

- **P** = `½ · [Ks, -Ks; -Ks, Ks]` so that osqp's `½ uᵀPu` evaluates to
  `¼ βᵀKsβ`, matching the `−¼` coefficient in Theorem 2.

- **q** = `[y(ε/100 − 1); y(1 + ε/100)]` (identical to Model 1)

- **Equality:** `[1ᵀ, −1ᵀ] u = 0`

- **Box:** `0 ≤ αk ≤ 100C/yk`, `0 ≤ αk* ≤ 100C/yk` (identical to Model
  1)

Note: `Ks = Ω + a·Ω*` carries **no** `½` factor here. The `½` lives in P
so that osqp's internal `½` produces the required `¼` overall. Contrast
with
[`rmspe_sym_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_sym_lssvr.md)
(Model 4), which uses
[`sym_kernel_matrix()`](https://pbenavidesh.github.io/psvr/reference/sym_kernel_matrix.md)
returning `½(Ω + a·Ω*)` = `Ωs`.

The kernel must satisfy Assumption 3 of the paper; see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

## Examples

``` r
X <- matrix(1:6, ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- mape_sym_svr(X, y, kernel = K, C = 10, eps = 5, a = 1)
predict(fit, X)
#> [1] 2.205 3.990 5.890
```
