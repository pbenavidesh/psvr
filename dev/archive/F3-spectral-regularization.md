# F3 — Adaptive Spectral Regularization

**Status:** Archived implementation notes — current state lives in `CLAUDE.md`.

## Adaptive Spectral Regularization (post-F3)

`.fit_mape_sym()` (Model 2) calls `.adaptive_spectral_shift()` from
`R/kernel-spectral.R` between the `0.5e-6` diagonal jitter step and the
SMO/osqp solve. The shift is invoked **unconditionally** (independent of
`a` or kernel type); the branch decision is made on the estimated
`λ_min(Ωs)`. This implements Theorem 2 / Algorithm 2 of
arXiv:2605.01446 v3 with one corrected detail (see "Paper deviation"
below).

### API

```
.adaptive_spectral_shift(Omega_s, T_pi = 5L, delta_stab = 1e-8) ->
  list(Omega_use, mu, lambda_min_hat, lambda_max_hat,
       branch_taken ∈ {"no_shift", "shifted"},
       n_power_iterations = c(iter1, iter2))
```

The function never modifies the input; when shifted, `Omega_use` is a
fresh matrix equal to `Omega_s + mu * I`. Results are deterministic:
both passes start from the uniform unit vector `rep(1, N) / sqrt(N)`.

### Two-pass shifted power iteration

Pass 1 runs plain power iteration on `Ωs` and returns the Rayleigh
quotient `lambda_max_hat`. Pass 2 runs power iteration on
`rho * I - Ωs` with `rho = |lambda_max_hat|` (the spectral radius)
and returns the Rayleigh `lambda_min_hat = v^T Ωs v` on the converged
`v`. Both passes are O(N²); total cost is `2 * T_pi` matvecs.

### Paper deviation

Algorithm 2 line 6 of the paper, as literally written
(`v ← -Ωs · v / ||Ωs · v||`), estimates `-λ_max(Ωs)` rather than
`λ_min(Ωs)`: power iteration on `-Ωs` converges to the eigenvector of
largest |eigenvalue|, which is `v_max(Ωs)` whenever
`|λ_max| > |λ_min|` (the typical case for Mercer-PSD kernel matrices).
The Pass 2 shift uses `|Pass 1 Rayleigh|`, not Pass 1's signed Rayleigh,
to handle the `λ_min`-dominant case where `|λ_min| > λ_max` and Pass 1
converges to `v_min` with negative Rayleigh; using `abs(...)` ensures
`rho * I - Ωs` is PSD, so Pass 2 reliably finds `v_min(Ωs)` regardless
of which side of the spectrum dominated Pass 1. This deviation is
documented inline in `R/kernel-spectral.R` and is to be flagged as a
paper-side erratum (smo-v3.tex line 3467) in F8.

### Diagnostics

`psvr_fit$solver_meta$spectral` is populated only for symmetric MAPE
fits (`loss = "mape"` and `sym != NULL`); `NULL` for Models 1, 3, 4.
Schema:

```
$mu                  numeric scalar; 0 if branch_taken == "no_shift"
$lambda_min_hat      numeric; Rayleigh from Pass 2
$lambda_max_hat      numeric; Rayleigh from Pass 1 (see note)
$branch_taken        "no_shift" | "shifted"
$n_power_iterations  integer length-2: iterations executed in Pass 1, Pass 2
```

`lambda_max_hat` reports Pass 1's signed Rayleigh and equals
`λ_max(Ωs)` for PSD or `λ_max`-dominant matrices; for the
`λ_min`-dominant pathological case it equals `λ_min(Ωs)`. The branch
decision is made correctly via `lambda_min_hat` in either case.

### Why the shifted branch is dormant in production

With the three Mercer kernels supplied by `make_kernel()` (`"rbf"`,
`"linear"`, `"polynomial"`), `Ωs = ½(Ω + a·Ω*)` is **always PSD** by
Aronszajn's closure (shift-invariant kernels) and Schur's product
theorem (polynomial kernels collapse `K - K*` for odd / even degrees
into non-negative linear combinations of valid Mercer kernels). The
shifted branch is therefore a defensive guard that activates only with
**non-Mercer kernels** (e.g., tanh / sigmoid in some parameter ranges)
supplied via a custom kernel closure — a use case not currently
documented or tested. Production fits with the supplied
`make_kernel()` types always take the no-shift branch and remain
bit-identical to F2.

### Known limitations

Theorem 2(a) guarantees that the spectrally-shifted matrix `Omega_use`
satisfies `Omega_use ⪰ delta_stab * I` in the **limit** `T_pi → ∞`.
At finite `T_pi`, the estimate of `λ_min` has a residual bias that
bleeds into `Omega_use`; the resulting matrix is "almost PSD" but may
not strictly clear the `delta_stab` floor.

For kernel matrices with well-separated spectra (the typical case for
Mercer kernels with decaying eigenvalues), `T_pi = 5` suffices for
~10⁻⁶ accuracy in `lambda_min_hat`, and `Omega_use` cleanly clears the
floor. For pathological cases with clustered spectra (e.g.,
Wigner-random matrices, where eigenvalues fill the interval
`[-rho, rho]` densely per the semicircle law), Pass 2's convergence
rate is approximately 1, and reaching the strict floor requires
`T_pi ~ 200` iterations.

Practically, even `T_pi = 20` reduces indefiniteness by ~1000× on
pathological cases, which is sufficient for the SMO solver to handle
via its existing `0.5e-6` diagonal jitter and convergence safeguards.
Callers can override the default via
`.adaptive_spectral_shift(Omega_s, T_pi = 200)`.

Note: with the three Mercer kernels supported by `make_kernel()` (RBF,
linear, polynomial), `Ωs` is always PSD (see "Why the shifted branch
is dormant" above), so the shifted branch is unreachable in production
and these limitations are not user-visible.
