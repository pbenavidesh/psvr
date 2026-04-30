# Fit symmetric LS-SVR with RMSPE loss (Model 4)

Solves the linear system derived in Theorem 4 of Benavides-Herrera et
al. (2026). Identical in structure to
[`rmspe_lssvr()`](https://pbenavidesh.github.io/psvr/reference/rmspe_lssvr.md)
(Model 3) but replaces the kernel matrix Ω with the symmetrized matrix
`Ωs = ½(Ω + a·Ω*)`, where `Ω*ₖₗ = K(xₖ, -xₗ)`. The system solved is

## Usage

``` r
rmspe_sym_lssvr(X, y, kernel, gamma, a = 1, precondition = "auto")
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

- precondition:

  Optional symmetric rescaling preconditioner derived from Remark 17 of
  the companion paper, used when the target dynamic range
  `ρ = max(y) / min(y)` is large enough to make `Ωs + YΓ`
  ill-conditioned. One of:

  `"auto"` (default)

  :   Apply the preconditioner when `ρ > 10`.

  `"always"`

  :   Apply the preconditioner unconditionally.

  `"never"`

  :   Disable the preconditioner (legacy behaviour).

  a positive numeric scalar

  :   Apply when `ρ > precondition`.

## Value

An object of class `"psvr_rmspe_sym"`, a list with components:

- `alpha`:

  Numeric vector of dual variables (length N), in the original variable
  space.

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

- `precondition_applied`:

  Logical scalar; `TRUE` if the preconditioner was applied for this fit.

## Details

    [ 0   1ᵀ      ] [ b ]   [ 0 ]
    [ 1   Ωs + YΓ ] [ α ] = [ y ]

where `YΓ = diag(y₁²/Γ, …, yN²/Γ)`.

The kernel must satisfy Assumption 3 of the paper (kernel symmetry):
`K(-xi, xj) = K(xi, -xj)` and `K(-xi, -xj) = K(xi, xj)`. RBF and
even-degree polynomial kernels satisfy this; see
[`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md).

When `ρ = max(y) / min(y)` is large, `Ωs + YΓ` becomes ill-conditioned
because the diagonal of `YΓ = diag(yₖ²/Γ)` varies as `O(ρ²)`. Remark 17
of the companion paper derives a symmetric rescaling preconditioner
`P = diag(1/yₖ)` via the change of variable `α = P ᾱ` (i.e.
`αₖ = ᾱₖ / yₖ`). Multiplying the inner block of the bordered system by
`P` from the left gives
`(P Ωs P + Γ⁻¹·I) ᾱ = P y − b · P 1 = 1 − b · P 1`, with
constant-diagonal regularization `Γ⁻¹ · I`. The preconditioner is
applied to the symmetrized kernel matrix `Ωs` (after symmetrization).
The constraint `1ᵀ α = 0` becomes `(P 1)ᵀ ᾱ = 0`, so the bordered system
used by the preconditioned solver is


    [ 0      (P 1)ᵀ          ] [ b ]   [ 0 ]
    [ P 1    P Ωs P + Γ⁻¹·I   ] [ ᾱ ] = [ 1 ]

Recovery is `α = ᾱ / y` (elementwise division). The bias `b` is the same
constraint multiplier in both systems.

This is a strict change of variable: in exact arithmetic the
preconditioned and unconditioned solvers produce identical predictions.
Its purpose is to preserve solver accuracy under finite floating-point
precision when `ρ` is large; for moderate `ρ` the two paths agree to
within machine epsilon. Use `precondition = "auto"` (default) for
typical workloads, `"never"` for legacy behaviour, or a custom numeric
threshold for fine-grained control. The chosen behaviour is recorded in
`precondition_applied`.

## Examples

``` r
X <- matrix(c(1, 2, 3, 4, 5, 6), ncol = 2)
y <- c(2.1, 3.8, 6.2)
K <- make_kernel("rbf", sigma = 1)
fit <- rmspe_sym_lssvr(X, y, kernel = K, gamma = 1, a = 1)
predict(fit, X)
#> [1] 2.769355 2.853120 2.886170
```
