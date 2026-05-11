# psvr 0.0.2.9004 (development)

## Breaking changes

* `psvr_fit$alpha` for MAPE models is renamed to `psvr_fit$beta` to
  reflect its mathematical role (β = α − α*, length `n_sv`,
  post-pruning). The new `psvr_fit$alpha` and `psvr_fit$alpha_star`
  (length `N`, pre-pruning) expose the true SMO dual variables α and α*,
  required for the warm-start API of Theorem 5 (arXiv:2605.01446 v3).
  LS-SVR models (`loss = "rmspe"`) retain previous semantics:
  `fit$alpha` is the linear-system solution, `fit$alpha_star = NULL`,
  `fit$beta = NULL`. User code reading `fit$alpha` from MAPE fits must
  be updated to `fit$beta` for the post-pruned coefficient vector.

## New features

* `psvr()` now accepts `alpha_init`, `alpha_star_init`, and
  `warm_start_check` for SMO warm-start (`loss = "mape"` only).
  Validation: length-`N` vectors, strictly positive `C_k`; Algorithm 1
  projection applied to ensure feasibility regardless of input.

* New `psvr_cv()` helper for cross-validation with automatic warm-start
  across folds. Accepts an `rsample::rset` (`vfold_cv`, `mc_cv`, etc.)
  or a list of split tuples. Returns a tibble with per-fold fits,
  predictions, metrics, iter counts, and elapsed times.

## Internal changes

* `psvr_fit$solver_meta` now propagates `iters` and `converged` from the
  SMO solver (previously hard-coded to `NA`).

* `R/warm_start.R` implements Algorithm 1 of arXiv:2605.01446 v3 with a
  paper-text deviation in Step 2: distribute the equality-constraint
  violation over the new-sample subset (`S_new \ S_prev`) rather than
  uniformly over all `N`. Preserves retained-sample values to `1e-12`.
  See CLAUDE.md "Warm-start API" for rationale; paper TODO #6 for
  incorporation into smo-v3.tex Algorithm 1.

## Empirical findings

* **T5 warm-start cumulative speedup is approximately linear in fold
  count, not exponential.** On 10-fold CV: 1.12× at N=300 (ρ_y=2388),
  1.14× at N=1000 (ρ_y=16265). Per-fold `T_warm / T_cold ≈ 0.88`, not
  the 0.2 implicit in the paper's linear-convergence model. Paper
  TODO #7 flags this for recalibration of the smo-v3.tex Theorem 5
  prediction.

* **Working set selection evaluation: Fan-Chen-Lin WSS3 (libsvm-style)
  is sufficient for MAPE-SVR.** During F5 development we evaluated two
  alternatives — the saturation-distance multiplier from
  arXiv:2605.01446 v3 Theorem 4 and the Glasmachers-Igel (2006)
  maximum-gain WSS — and found neither provides empirical benefit on
  MAPE-SVR's heterogeneous-`C_k` regime. The "saturation problem" both
  were targeting is a phantom: WSS3's analytic step `δ_unc` typically
  fits the per-sample box without clipping. Paper TODO #5 flags this
  for restructure of smo-v3.tex Section 6 / Theorem 4 (drop entirely).

# psvr 0.0.2.9003 (development)

## Internal changes

* `.smo_solve()` (`R/smo_solve.R`) now implements Theorems 3 and 8 of
  arXiv:2605.01446 v3:

  - **Theorem 3 (asymmetric freeze):** the uniform freeze-counter
    threshold `n_freeze = 5L` is replaced by per-sample, per-variable-type
    thresholds:

        n_freeze_astar_per[k] = max(1L, floor(n_freeze * y[k] / mean(y)))
        n_freeze_alpha_per[k] = max(5L, ceil( n_freeze * mean(y) / y[k]))

    α*-variables tied to high-`y_k` samples freeze faster; α-variables
    tied to high-`y_k` samples freeze slower. Reduces to the scalar
    default when `y` is homogeneous.

  - **Theorem 8 (per-pair tolerance):** the uniform convergence threshold
    `tol * mean(y)` is replaced at the convergence test by a per-pair
    threshold `tol * max(y[p], y[k_j_w1])`, where `(p, k_j_w1)` is the
    WSS1 pair. The WSS3 candidate filter at the working-set selection
    step retains the global `tol * mean(y)` for noise-floor purposes.

## Behavior change

* SMO iteration count decreases by ~10–30% on heterogeneous-target
  datasets (`rho_y = max(y) / min(y) >= 50`). The wall-clock speedup
  is smaller (depends on `N`: kernel-matrix construction overhead
  dominates at moderate `N`). Empirical benchmark at `N = 200`,
  `rho_y = 1273`: 24.7% iter reduction, 5.6% wall-clock speedup.
  Homogeneous datasets (`rho_y ~ 1`) are unaffected — the new
  thresholds collapse to the F3 defaults.

## Numerical drift

* Predictions on heterogeneous-target datasets differ from v0.0.2.9002
  by `O(tol * max(y))`. Empirical magnitude on the snapshot fixture
  (`rho_y = 6.7`, `max(y) = 2.6`, `tol = 1e-3`): max drift
  `8.5e-4` per prediction (well below the per-pair tolerance floor of
  `2.6e-3`).
* The 28 golden snapshot tests have been re-recorded with F4 numerics
  as the new regression baseline at tolerance `1e-10` within F4.
  Bit-identicalidad with v0.0.2.9002 is intentionally NOT preserved on
  SMO-backed paths; LS-SVR paths (Models 3, 4 via linear system)
  remain bit-identical.

## Paper deviation

* The convergence test uses the WSS1 pair `(i_w1, j_w1)` rather than
  the paper's literal text ("j* = WSS3 pick"). Rationale documented
  inline in `R/smo_solve.R`: `Delta_w3 <= Delta_w1` by construction,
  so testing `Delta_w3` against the tolerance would stop prematurely
  before the true KKT optimality gap is below tolerance. Flagged for
  paper-side notation fix in F8 (paper TODO #4).

# psvr 0.0.2.9002 (development)

## Internal changes

* `.fit_mape_sym()` (Model 2) now invokes Algorithm 2 (Theorem 2 of
  arXiv:2605.01446 v3) via the new `.adaptive_spectral_shift()` in
  `R/kernel-spectral.R`. Two-pass shifted power iteration estimates
  `λ_min(Ωs)`; if numerically negative, an adaptive `μ·I` shift is
  applied so the SMO Hessian is provably PSD. Diagnostics (`mu`,
  `lambda_min_hat`, `lambda_max_hat`, `branch_taken`,
  `n_power_iterations`) surface under
  `psvr_fit$solver_meta$spectral` for symmetric MAPE fits; `NULL` for
  Models 1, 3, 4. The implementation deviates from the paper's
  Algorithm 2 line 6 by using the spectral radius `|Pass 1 Rayleigh|`
  rather than the signed Rayleigh as the Pass 2 shift; rationale is
  documented inline in `R/kernel-spectral.R` and to be flagged as a
  paper-side erratum in F8. With the three Mercer kernels supported by
  `make_kernel()` (RBF / linear / polynomial), `Ωs` is always PSD by
  Aronszajn closure / Schur product theorem, so the shifted branch is
  unreachable in production fits and predictions are bit-identical to
  v0.0.2.9001 at tolerance `1e-10` on the 16 + 12 golden snapshots.

# psvr 0.0.2.9001 (development)

## Internal changes

* The SMO solver now reads kernel values via a `.make_kernel_accessor()`
  wrapper (`R/kernel-accessor.R`), decoupling the inner loop from the
  matrix representation. The F2 implementation is a thin closure over
  the materialised `Ω`/`Ωs`; future phases (F3 spectral shift, F6 LRU
  cache, F7 block working set) will swap in alternative implementations
  without touching the SMO loop or the fitters. No user-visible change;
  predictions are bit-identical to v0.0.2.9000 at tolerance `1e-10` on
  the 16 + 12 golden snapshots.

# psvr 0.0.2.9000

## New features

* New unified entry point `psvr(X, y, loss, sym, kernel, ...)`. Selects
  among the four model families via `loss = "mape" | "rmspe"` and
  `sym = NULL | +1L | -1L`. Returns a single `psvr_fit` S3 object with a
  unified field schema (`alpha`, `b`, `support_data`, `support_targets`,
  `n_train`, `n_sv`, `p_train`, `hyperparameters`, `solver_meta`, plus
  `loss`, `sym`, `kernel`).
* New methods on the `psvr_fit` class: `predict()`, `print()`, `coef()`,
  and `summary()`. The summary method is new — there was no analogue on
  the four legacy classes.

## Deprecations

* `mape_svr()`, `mape_sym_svr()`, `rmspe_lssvr()`, `rmspe_sym_lssvr()`
  are soft-deprecated. They continue to work and return the same legacy
  shape (`psvr_mape`, `psvr_mape_sym`, `psvr_rmspe`, `psvr_rmspe_sym`)
  with the same predict/print/coef methods, but each emits
  `.Deprecated("psvr")`. Migrate by replacing, for example,
  `mape_svr(X, y, kernel = K, C = 10, eps = 5)` with
  `psvr(X, y, loss = "mape", kernel = K, C = 10, eps = 5)`. Scheduled
  removal: v0.2.0 or later.

## Internal changes

* The 12 parsnip fit wrappers (`psvr_mape_rbf_fit()` etc.) are now
  tagged `@keywords internal` and hidden from the pkgdown reference
  index. They remain exported (parsnip's resolver requires it) but are
  no longer advertised as user API; call [psvr()] for direct fitting.
  The 12 spec constructors (`psvr_mape_rbf()` etc.) remain public API
  and are unchanged.
* The four model fitter bodies have been renamed to internal helpers
  (`.fit_mape()`, `.fit_mape_sym()`, `.fit_rmspe()`, `.fit_rmspe_sym()`).
  The deprecation wrappers delegate to these directly.
* DRY consolidation: shared validation, large-N warnings, predict
  dispatch, preconditioner resolution, and kernel description live in
  new internal-only files under `R/utils-*.R`.
* Model 2 (symmetric MAPE epsilon-SVR) now uses `sym_kernel_matrix()`
  to build `Ωs = ½(Ω + a·Ω*)` (matching Model 4's existing convention),
  with diagonal jitter `0.5e-6` to preserve bit-identicality with the
  pre-F1 path.

# psvr 0.0.2 (2026-04-30)

## New features

* The `precondition` argument from `rmspe_lssvr()` and `rmspe_sym_lssvr()`
  is now configurable via parsnip's `set_engine()` for all six RMSPE spec
  functions (rbf/poly/linear, both symmetric and non-symmetric). Default
  is `"auto"`; pass via `set_engine("psvr", precondition = "always")`,
  `"never"`, or a numeric threshold. Not registered as a tunable
  parameter — it is a configuration flag, not a hyperparameter.

# psvr 0.0.1

## New features

* Added optional preconditioner for LS-SVR variants (`rmspe_lssvr()`,
  `rmspe_sym_lssvr()`) via the `precondition` argument. Accepted values:
  "always", "never", "auto" (default; activates when
  max(y)/min(y) > 10), or a numeric threshold. The preconditioner is a
  mathematically exact change of variable that improves numerical
  conditioning at large target dynamic ranges without changing
  predictions in exact arithmetic.
* Returned model objects now include `precondition_applied` (logical)
  for diagnostic transparency. The `print` method displays this field
  when TRUE.

## Documentation

* Updated `@details` for both LS-SVR fitters with the bordered-system
  formulation and the recovery formula α_k = α̃_k / y_k.

# psvr 0.0.0.9005

### New features

- `cost_psvr_ls_data(y, n, width_log2)` — a data-driven cost range
  for LS-SVR psvr models (`m3`, `m4`). Default upper bound scales
  as `log2(var(y) * N) + width_log2` (default `width_log2 = 4`),
  the standard heuristic for the LS-SVR regularisation parameter
  `Γ` (Suykens et al. 2002, *Least Squares Support Vector
  Machines*, §3.1.3). Use this instead of `cost_psvr()` for
  `m3`/`m4` workflows; the static `[-2, 10]` log2 range
  underestimates `Γ` on benchmark datasets (Boston Housing optimum
  `Γ ≈ 1.7 × 10⁴`, vs. `cost_psvr()` upper bound `2^10 = 1024`).
- `psvr_option_add_cost_ls(wf_set, y, ...)` — a convenience wrapper
  that applies `cost_psvr_ls_data()` to every LS-SVR psvr workflow
  in a workflow set (those whose `wflow_id` matches `m3|m4`),
  analogous to `psvr_option_add()` for `rbf_sigma`.

# psvr 0.0.0.9004

### New features

- `sym_type` is now a tunable parsnip argument in
  `psvr_mape_sym_rbf()` and `psvr_rmspe_sym_rbf()`. Pass
  `sym_type = tune()` to let CV select between even (a = 1)
  and odd (a = -1) symmetry automatically.
- Engine default `a = 1L` removed from symmetric RBF specs; symmetry
  type is now controlled exclusively through the `sym_type` model
  argument.

# psvr 0.0.0.9003

## UX improvements

* New function `rbf_sigma_psvr_data()`: combines `sigma_heuristic()` and
  `rbf_sigma_psvr()` in a single call, returning a dials parameter with a
  data-driven search range centred on the median pairwise distance.

* New function `psvr_option_add()`: applies `option_add()` to every psvr
  workflow (those whose `wflow_id` matches `m1|m2|m3|m4`) in a workflow set
  simultaneously, replacing the `rbf_sigma` parameter with a data-driven one.
  Replaces four individual `option_add()` calls in typical workflows.

* Symmetric model specs (`psvr_mape_sym_*`, `psvr_rmspe_sym_*`): even symmetry
  (`a = 1L`) is now the default engine argument. Calling `set_engine("psvr")`
  without `a = 1L` now produces the same result as before — no action required
  for existing code that already passed `a = 1L`.

* All four core fitting functions now emit a warning when `N > 2000` to
  remind users that the O(n²) kernel matrix may be expensive.

* The `y > 0` input check in all four core fitting functions now reports the
  number of offending values and their minimum, making it easier to diagnose
  data issues.

# psvr 0.0.0.9002

## User-friendly hyperparameter defaults

* New function `margin_percentage()`: a dials parameter for the epsilon tube
  half-width in MAPE models, expressed in percentage units with default range
  `[1, 20]`. Replaces the absolute-unit `dials::svm_margin()` for all 6 MAPE
  specs.

* New function `sigma_heuristic()`: returns the median pairwise Euclidean
  distance of a predictor matrix — a standard data-driven starting point for
  the RBF kernel bandwidth (Schölkopf & Smola, 2002).

* New function `rbf_sigma_psvr()`: a dials parameter for the RBF kernel
  bandwidth with a `finalize` hook that auto-sets the search range using
  `sigma_heuristic()` when training data are available. Replaces
  `dials::rbf_sigma()` for all RBF specs.

* New function `cost_psvr()`: a dials parameter for the regularisation
  parameter with range `[-2, 10]` on the log2 scale (approx. 0.25 to 1024),
  wider than `dials::cost()` to accommodate the larger values typically needed
  by LS-SVR models. Replaces `dials::cost()` for all 12 specs.

* `rbf_sigma_psvr()`: removed non-functional `finalize` hook. The search
  range must be set manually using `sigma_heuristic()` and `option_add()`.
  See `?rbf_sigma_psvr` for the recommended workflow.

# psvr 0.0.0.9001

## tidymodels / parsnip integration — breaking change

Expanded parsnip integration from 4 to 12 model specs following the tidymodels
pattern used by `svm_rbf()`, `svm_poly()`, and `svm_linear()`:

* `psvr_mape_rbf()`, `psvr_mape_poly()`, `psvr_mape_linear()` — Model 1 specs.
* `psvr_mape_sym_rbf()`, `psvr_mape_sym_poly()`, `psvr_mape_sym_linear()` — Model 2 specs.
* `psvr_rmspe_rbf()`, `psvr_rmspe_poly()`, `psvr_rmspe_linear()` — Model 3 specs.
* `psvr_rmspe_sym_rbf()`, `psvr_rmspe_sym_poly()`, `psvr_rmspe_sym_linear()` — Model 4 specs.

Kernel parameters are now tunable parsnip args mapped to existing dials params:
`rbf_sigma` → `dials::rbf_sigma()`, `degree` → `dials::degree()`,
`scale_factor` → `dials::scale_factor()`.  The kernel closure is built inside
each fit wrapper and no longer appears in the parsnip layer.

The old single-spec API (`psvr_mape()`, `psvr_mape_sym()`, `psvr_rmspe()`,
`psvr_rmspe_sym()`) has been removed.  Migrate by replacing, for example,
`psvr_rmspe(cost = tune()) |> set_engine("psvr", kernel = K)` with
`psvr_rmspe_rbf(cost = tune(), rbf_sigma = 1) |> set_engine("psvr")`.

## Documentation and testing

* All vignettes and pkgdown articles updated to the new spec-based API.
* 12 new smoke tests (fit + predict) for all specs in
  `tests/testthat/test-parsnip.R`.
* pkgdown articles reorganised into three named groups: *Get Started*,
  *Case Studies*, and *Technical Notes*.

# psvr 0.0.0.9000

Initial development release.

## New models

* `mape_svr()` — epsilon-SVR with MAPE loss (Model 1). Solves the dual QP via
  `osqp` with per-sample box constraints `|βk| ≤ 100C/yₖ`.

* `mape_sym_svr()` — symmetric epsilon-SVR with MAPE loss (Model 2). Enforces
  `f(x) = a·f(-x)` by replacing the kernel with the symmetric kernel
  `Ks(xi, xj) = K(xi, xj) + a·K(xi, -xj)`.

* `rmspe_lssvr()` — LS-SVR with RMSPE loss (Model 3). Solves the (N+1)×(N+1)
  linear system directly via `base::solve()`.

* `rmspe_sym_lssvr()` — symmetric LS-SVR with RMSPE loss (Model 4). Same
  linear system as Model 3 with the symmetrized kernel matrix
  `Ωs = ½(Ω + a·Ω*)`.

All four models are derived from a unified percentage-error loss framework;
see Benavides-Herrera et al. (2026) for the mathematical proofs.

## Kernel interface

* `make_kernel()` — factory returning a kernel closure from `type` ∈
  `{"rbf", "linear", "polynomial"}`. RBF and even-degree polynomial kernels
  satisfy the symmetry assumption (Assumption 3) required by Models 2 and 4.

## tidymodels / parsnip integration

Four parsnip model specifications for use within tidymodels workflows:

* `psvr_mape()` — spec for Model 1; hyperparameters `cost` and `svm_margin`.
* `psvr_mape_sym()` — spec for Model 2; hyperparameters `cost` and `svm_margin`.
* `psvr_rmspe()` — spec for Model 3; hyperparameter `cost` (maps to `Γ`).
* `psvr_rmspe_sym()` — spec for Model 4; hyperparameter `cost` (maps to `Γ`).

The kernel and (for symmetric models) symmetry parameter `a` are engine
arguments supplied via `set_engine("psvr", kernel = ..., a = ...)`.
Hyperparameters map to `dials::cost()` and `dials::svm_margin()` for
compatibility with `tune_grid()`.

## Testing

* 84 unit tests covering all four models: kernel correctness, QP/linear-system
  solutions, support vector selection, bias recovery, predict consistency,
  and input validation.
