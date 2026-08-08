# F5 — Warm-start API + Working Set Selection Evaluation

**Status:** Archived implementation notes — current state lives in `CLAUDE.md`.

## Warm-start API + Working Set Selection Evaluation (post-F5)

### Working Set Selection: empirical finding

`.smo_solve()` uses Fan-Chen-Lin (2005, JMLR 6:1889–1918) WSS3 with the
unconstrained-gain score `(τ_i − τ_j)² / η`. During F5 development we
evaluated two alternatives:

- The saturation-distance multiplier proposed in arXiv:2605.01446 v3
  Theorem 4 (with tunable `alpha_wss` parameter): empirically failed on
  MAPE-SVR. With `alpha_wss = 0.5` (paper default), N=200 ρ_y=1273 RBF
  bench showed 140% iter **increase** vs unmodified WSS3. Math invariants
  held (strict descent OK, converged to same KKT optimum), but the
  multiplier penalized perfectly-good candidates whose only "fault" was
  small `R_j` relative to a mean computed over the active set. Root
  cause: the heuristic assumed room-poor candidates are a minority, but
  in MAPE-SVR's heterogeneous `C_k = 100C/y_k` regime, room-poor
  candidates dominate the active set during the early-to-mid SMO phase.

- Glasmachers-Igel (2006, JMLR 7:1437–1466) maximum-gain WSS3 with exact
  box-clipped realized gain: empirically equivalent to WSS3 on tested
  regimes (N=50/200, ρ_y=6.7–1273, RBF). Iter counts matched
  bit-exactly (137=137, 202=202, 43=43 snapshot fixture; 280=280 bench).
  Max-gain incurred ~12% per-iteration wall overhead from the additional
  clipping arithmetic. Root cause for the equivalence: WSS3's analytic
  step `δ_unc = gap / η` typically fits within the per-sample box bounds
  even for highly heterogeneous `C_k`, so clipping rarely activates and
  max-gain reduces to WSS3 numerically.

Conclusion: the "saturation problem" both the heuristic and max-gain
were attempting to solve is a phantom at the WSS3 selection level for
MAPE-SVR. Fan-Chen-Lin's standard libsvm WSS3 is sufficient. This is a
useful negative result for future MAPE-SVR optimization research. See
paper TODO #5 for the implication on smo-v3.tex Theorem 4.

### Warm-start API (Theorem 5 with paper deviation)

The package provides direct warm-start via the `alpha_init` and
`alpha_star_init` parameters of `psvr()` (validated to length N,
strictly positive targets for MAPE loss). The `.warm_start_init()`
helper in `R/warm_start.R` implements Algorithm 1 of arXiv:2605.01446
v3 with one deviation from the paper text:

- **Paper Algorithm 1 Step 2:** uniform shift over ALL `N` samples by
  `violation / N`. Empirically degrades the warm-start advantage on
  10-fold CV (Round 1: 0.97× cumulative).
- **Our deviation:** distribute violation only over the new-sample
  subset (`S_new \ S_prev`). Rationale: the equality-constraint
  violation arises entirely from removed samples (`S_prev \ S_new`)
  whose dual values are no longer used; retained samples
  (`S_prev ∩ S_new`) were at the equality-constraint manifold at the
  previous fold's optimum and should be preserved exactly. The
  new-sample-only projection preserves retained values to `1e-12`
  (test verified). When the per-new shift forces clipping at `0` or
  `C_k`, a one-pass uniform refinement absorbs the residual (rare in
  typical CV).

This deviation is documented as paper TODO #6 for incorporation into
the final paper text.

### CV helper `psvr_cv()`

`psvr_cv(splits, X_var, y_var, ...)` accepts an `rsample::rset` OR a
list-of-tuples and orchestrates warm-start across folds using
row-ID-based `new_mask` inference. Returns a plain tibble with
`split_id`, `fit`, `predictions`, `metrics`, `iter_count`,
`elapsed_sec`, `warm_started`.

Scope: A′ (`psvr_cv` as explicit helper, no parsnip auto-warm-start in
`tune_grid`). F5b will add parsnip integration if/when warranted.

### Empirical speedup calibration

Paper-predicted cumulative speedup: **3–7×** on 10-fold CV with
linear-convergence assumption (`T_warm / T_cold ≈ 0.2` per fold).

Observed:

- N=300 (ρ_y=2388, RBF, 10-fold): **1.12× wall**, 12.7% iter reduction.
- N=1000 (ρ_y=16265, RBF, 10-fold): **1.14× wall**, 13.8% iter reduction.

Per-fold `T_warm / T_cold ≈ 0.88` (not 0.2). Cumulative speedup is
approximately linear in fold count, not exponential. This is
N-independent at our tested regimes — see paper TODO #7 for the
recalibration recommendation.

### Breaking-change: `fit$alpha` → `fit$beta` for MAPE

For MAPE fits (`psvr_fit` with `loss = "mape"`), the field formerly
called `fit$alpha` (length `n_sv`, post-pruning, holding `β = α − α*`)
is renamed to `fit$beta`. Two new length-`N` (pre-pruning) fields
`fit$alpha` and `fit$alpha_star` expose the true SMO dual variables —
required as warm-start state by `psvr_cv()`. LS-SVR fits
(`loss = "rmspe"`) retain previous semantics: `fit$alpha` is the
linear-system solution, `fit$alpha_star = NULL`, `fit$beta = NULL`.
Downstream code reading `fit$alpha` from MAPE fits must switch to
`fit$beta`.

`psvr_fit$solver_meta` now propagates real `iters` and `converged`
values from the SMO solver (previously hard-coded to `NA`).
