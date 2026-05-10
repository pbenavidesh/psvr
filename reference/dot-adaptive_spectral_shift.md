# Adaptive spectral regularization (internal, F3)

Implements Theorem 2 of arXiv:2605.01446 v3 with corrected eigenvalue
estimator. Adds a `mu * I` shift to `Omega_s` when its smallest
eigenvalue is numerically negative, so the SMO Hessian is provably PSD
(`>= delta_stab * I`). Returns the matrix untouched when already PSD.

## Usage

``` r
.adaptive_spectral_shift(Omega_s, T_pi = 5L, delta_stab = 1e-08)
```

## Arguments

- Omega_s:

  Symmetrized kernel matrix `Omega_s = 1/2 (Omega + a * Omega^*)`,
  already with any caller-applied jitter.

- T_pi:

  Power-iteration steps per pass (default `5L`).

- delta_stab:

  Numerical PSD floor (default `1e-8`).

## Value

A list with components:

- `Omega_use`:

  Matrix to pass to the SMO/QP solver.

- `mu`:

  Numeric scalar; the shift applied (`0` if no shift).

- `lambda_min_hat`:

  Numeric scalar; Rayleigh-quotient estimate of `lambda_min(Omega_s)`
  from Pass 2.

- `lambda_max_hat`:

  Numeric scalar; Pass 1's Rayleigh quotient. For PSD or
  `lambda_max`-dominant matrices (the typical case for Mercer kernels in
  this package), this equals `lambda_max(Omega_s)`. For
  `lambda_min`-dominant matrices (`|lambda_min| > lambda_max` — a
  pathological case not reachable with
  [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)-supplied
  kernels), it equals `lambda_min(Omega_s)`. The branch decision in
  either case is made correctly via the Pass 2 estimate of
  `lambda_min_hat`.

- `branch_taken`:

  Character `"no_shift"` or `"shifted"`.

- `n_power_iterations`:

  Integer vector of length 2; iterations executed in Pass 1 and Pass 2.

## Algorithm and paper deviation

Algorithm 2 line 6 of the paper, as literally written
(`v <- -Omega_s * v / ||Omega_s * v||`), estimates
`-lambda_max(Omega_s)` rather than `lambda_min(Omega_s)`: power
iteration on `-Omega_s` converges to the eigenvector of largest
\|eigenvalue\|, which is `v_max(Omega_s)` whenever
`|lambda_max| > |lambda_min|` (the typical case for Mercer-PSD or
near-PSD matrices). This implementation uses the standard two-pass
shifted power iteration: Pass 1 estimates `lambda_max` via power
iteration on `Omega_s`; Pass 2 estimates `lambda_min` via power
iteration on `rho * I - Omega_s` (whose dominant eigenvector is
`v_min(Omega_s)`). Both passes are O(N^2); the total cost is `2 * T_pi`
matvecs.

The Pass 2 shift uses the spectral radius `rho = |Pass 1 Rayleigh|`, not
Pass 1's signed Rayleigh. This handles the `lambda_min`-dominant
pathological case (`|lambda_min| > lambda_max`), where plain power
iteration on `Omega_s` in Pass 1 converges to `v_min` and gives a
negative Rayleigh; using `abs(Pass 1 Rayleigh)` ensures the shifted
operator `rho * I - Omega_s` is PSD, so Pass 2 reliably finds
`v_min(Omega_s)` regardless of which side of the spectrum dominated Pass
1.

## Determinism

Both passes start from the uniform unit vector `rep(1, N) / sqrt(N)`
(not random) so the routine is bit-reproducible.
