# Fit LS-SVR with RMSPE loss (Model 3)

Solves the linear system derived in Theorem 3 of Benavides-Herrera et
al. (2026). The primal objective is `½‖ω‖² + (Γ/2) Σ eₖ²/yₖ²`, leading
to the (N+1)×(N+1) system

## Usage

``` r
rmspe_lssvr(X, y, kernel, gamma, precondition = "auto")
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

- precondition:

  Optional symmetric rescaling preconditioner derived from Remark 17 of
  the companion paper, used when the target dynamic range
  `ρ = max(y) / min(y)` is large enough to make `Ω + YΓ`
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

An object of class `"psvr_rmspe"`, a list with components:

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

- `n_train`:

  Number of training observations.

- `p_train`:

  Number of training features (columns).

- `precondition_applied`:

  Logical scalar; `TRUE` if the preconditioner was applied for this fit.

## Details

    [ 0   1ᵀ     ] [ b ]   [ 0 ]
    [ 1   Ω + YΓ ] [ α ] = [ y ]

where `YΓ = diag(y₁²/Γ, …, yN²/Γ)` is added to the diagonal of Ω.

When the target dynamic range `ρ = max(y) / min(y)` is large, the
diagonal of `YΓ = diag(yₖ²/Γ)` varies as `O(ρ²)`, making `Ω + YΓ`
ill-conditioned. Remark 17 of the companion paper derives a symmetric
rescaling preconditioner `P = diag(1/yₖ)` via the change of variable
`α = P ᾱ` (i.e. `αₖ = ᾱₖ / yₖ`). Multiplying the inner block of the
bordered system by `P` from the left gives
`(P Ω P + Γ⁻¹·I) ᾱ = P y − b · P 1 = 1 − b · P 1`, with
constant-diagonal regularization `Γ⁻¹ · I` independent of `yₖ`. The
constraint `1ᵀ α = 0` becomes `(P 1)ᵀ ᾱ = 0`, so the bordered system
used by the preconditioned solver is

    [ 0      (P 1)ᵀ          ] [ b ]   [ 0 ]
    [ P 1    P Ω P + Γ⁻¹·I    ] [ ᾱ ] = [ 1 ]

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
fit <- rmspe_lssvr(X, y, kernel = K, gamma = 1)
predict(fit, X)
#> [1] 2.743967 2.904833 2.969809
```
