## Two-tier floating-point assertion policy (added 2026-08-09).
##
## Background. Until the first multi-platform CI run the package was only
## ever checked on one Windows x86_64 toolchain, where every R-vs-C++ parity
## assertion held bit-exactly. It does not hold off that machine:
##
##   * macOS aarch64 has a 64-bit `long double` (identical to `double`), so
##     the long-double accumulator discipline the R and C++ paths rely on to
##     agree exactly collapses there.
##   * The polynomial kernel differs by 1-2 ULP on every platform: R's `^`
##     uses repeated multiplication for integer exponents, `kernel_poly_cpp`
##     uses `powl`.
##
## Policy. The strict gates keep their original purpose — detecting
## intra-platform regressions across refactors — and are therefore run
## under `skip_on_cran()`, plus one CI job (`ubuntu-latest (release)`,
## which sets `NOT_CRAN: true`) that pins them on x86_64. Alongside each
## strict gate sits a tolerance assertion that runs unconditionally, so
## correctness is still verified on all five platforms.
##
## Cross-platform FP-noise tolerance.
## Chosen from the observed deviations in run 31291248498, not from a round
## number. The largest deviation among configurations where BOTH engines
## converged was 5.151e-14 (macOS aarch64, Model 2 MAPE-sym / RBF /
## bk4=TRUE, `alpha_star`); predictions on the same configs topped out at
## 3.730e-14, and the polynomial-kernel ULP differences sit near 2e-16
## relative. 1e-12 gives ~19x headroom over the worst observed value while
## remaining ~10 orders of magnitude tighter than the smallest genuine
## algorithmic divergence seen (6.1e-03). Values under test are O(1), so
## testthat's relative comparison is effectively absolute here.
PSVR_FP_TOL <- 1e-12

## Marginally-conditioned configuration tolerance.
## `Model 1 MAPE / poly_d3 / bk4=FALSE` converges on both engines but to
## different points: 6431 iterations on x86_64 (both engines, bit-equal)
## versus 6435 (R) / 7795 (Rcpp) on aarch64, with a 6.104e-03 maximum
## prediction difference. It sits on a working-set-selection near-tie that
## small FP differences flip.
##
## This cannot use PSVR_FP_TOL, and folding it into PSVR_FP_TOL would stop
## that constant asserting anything useful about the configs that do agree.
## It gets its own bound tied to the solver tolerance instead. The solver
## runs at `tol = 1e-3` and tests convergence per-pair against
## `tol_pair = tol * max(y[p], y[j])` (R/smo_solve.R:257); on this fixture
## y has mean 1.1360 and max 2.3300, so tol_pair <= 2.33e-03. The observed
## 6.104e-03 is 2.62x that bound and 5.4e-03 relative to mean(y) — the
## expected signature of two runs landing at different points inside the
## same tolerance ball, not of a defect.
##
## 1e-02 relative is ~4.3x the largest tol_pair and ~1.9x the observed
## deviation, yet ~20x below the smallest divergence produced by a config
## that genuinely failed to converge (2.146e-01). That gap is what makes it
## an assertion rather than a rubber stamp.
PSVR_MARGINAL_TOL <- 1e-2

## Pinned convergence map for the 16-config engine-equivalence matrix.
##
## Configurations that exhaust `max_iter` are the documented linear /
## polynomial MAPE-SVR pathology (CLAUDE.md "Known issues", paper TODO #5).
## Comparing two solvers that both failed to converge tests nothing, so the
## tolerance tier skips the value comparison for them.
##
## That skip is guarded. The set below is asserted to be EXACTLY the set of
## configs that hit the cap: a config outside it reaching `max_iter` fails
## (a regression that would otherwise hide behind a skip), and a config
## inside it converging also fails (an improvement that should force this
## list to be updated rather than silently widening the skip).
##
## Baseline measured on x86_64 (local Windows, devtools::load_all, all 16
## configs bit-equal across engines) and cross-checked against the macOS
## aarch64 diagnostics from run 31291248498. Note that `Model 2 MAPE-sym /
## linear` CONVERGES (43 / 23 iterations) — the pathology is not simply
## "every non-RBF kernel".
PSVR_MAXITER_CONFIGS <- c(
  "Model 1 MAPE / linear / bk4=FALSE",
  "Model 1 MAPE / linear / bk4=TRUE",
  "Model 1 MAPE / poly_d2 / bk4=FALSE",
  "Model 1 MAPE / poly_d2 / bk4=TRUE",
  "Model 2 MAPE-sym / poly_d2 / bk4=FALSE",
  "Model 2 MAPE-sym / poly_d2 / bk4=TRUE",
  "Model 2 MAPE-sym / poly_d3 / bk4=FALSE",
  "Model 2 MAPE-sym / poly_d3 / bk4=TRUE"
)

## Configs that converge on both engines but may land at different points.
PSVR_MARGINAL_CONFIGS <- c(
  "Model 1 MAPE / poly_d3 / bk4=FALSE"
)

## Memo cache so the strict and tolerance tiers can assert over the same
## computed values without paying to build them twice.
.psvr_memo_env <- new.env(parent = emptyenv())

psvr_memo <- function(key, expr) {
  hit <- .psvr_memo_env[[key]]
  if (!is.null(hit)) return(hit)
  val <- expr
  .psvr_memo_env[[key]] <- val
  val
}
