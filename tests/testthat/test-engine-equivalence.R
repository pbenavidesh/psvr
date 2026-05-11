## F7-C-full — engine="r" vs engine="rcpp" numerical equivalence canary.
##
## The strict bit-identicality gate for the C++ port: every fit configured
## (model × kernel × block_k4) must produce IDENTICAL doubles across the
## two engines. Predictions, solver_meta, alpha, alpha_star, b — all
## bit-equal. The 16-config matrix below is the regression surface that
## keeps the engines in lockstep through v0.0.4.x deprecation and v0.1.0
## removal of the R reference.
##
## Diagnostic policy: a bare expect_identical() gives binary pass/fail
## with no FP-level context. On any failure we print max diff, first
## differing indices, side-by-side values at full precision, plus iter
## and solver_meta deltas. This feeds the escalation policy directly
## (FP-order noise <1e-15 → investigate; BLAS divergence 1e-12 →
## investigate; algorithmic >1e-8 → urgent).

## ---- Fixture ----
make_eq_fixture <- function() {
  set.seed(2026)
  X <- matrix(stats::rnorm(50 * 5), 50, 5)
  y <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
  X_test <- matrix(stats::rnorm(20 * 5), 20, 5)
  list(X = X, y = y, X_test = X_test)
}

## ---- Diagnostic helper -----------------------------------------------------
## On failure, print:
##   * max elementwise prediction diff
##   * first 5 indices where diff > 0 with side-by-side values (digits=17)
##   * iter R vs Rcpp
##   * solver_meta field deltas: alpha, alpha_star, b, joint_updates,
##     k2_fallbacks, decoupling_rate, early/late_phase rates
.diagnose_engine_diff <- function(fit_r, fit_rcpp, preds_r, preds_rcpp,
                                  label = "") {
  delta_p <- abs(preds_rcpp - preds_r)
  cat(sprintf("\n--- ENGINE DIFF DIAGNOSTIC [%s] ---\n", label))
  cat(sprintf("  preds max diff : %.3e\n", max(delta_p)))
  first_5 <- utils::head(which(delta_p > 0), 5L)
  if (length(first_5) > 0L) {
    cat(sprintf("  first 5 diff indices: %s\n",
                paste(first_5, collapse = ", ")))
    cat(sprintf("    R    : %s\n",
                paste(format(preds_r[first_5],    digits = 17),
                      collapse = " ")))
    cat(sprintf("    Rcpp : %s\n",
                paste(format(preds_rcpp[first_5], digits = 17),
                      collapse = " ")))
  }
  cat(sprintf("  iters R=%d  Rcpp=%d  (Δ=%+d)\n",
              fit_r$solver_meta$iters, fit_rcpp$solver_meta$iters,
              fit_rcpp$solver_meta$iters - fit_r$solver_meta$iters))
  for (f in c("alpha", "alpha_star")) {
    v_r <- fit_r[[f]]; v_c <- fit_rcpp[[f]]
    if (!is.null(v_r) && !is.null(v_c) && length(v_r) == length(v_c)) {
      cat(sprintf("  %-12s max diff: %.3e\n", f, max(abs(v_c - v_r))))
    }
  }
  cat(sprintf("  b           R=%.17g  Rcpp=%.17g  diff=%.3e\n",
              fit_r$b, fit_rcpp$b, abs(fit_rcpp$b - fit_r$b)))
  for (f in c("joint_updates", "k2_fallbacks",
              "decoupling_rate",
              "early_phase_decoupling_rate",
              "late_phase_decoupling_rate")) {
    v_r <- fit_r$solver_meta[[f]]
    v_c <- fit_rcpp$solver_meta[[f]]
    cat(sprintf("  solver_meta$%-30s R=%s  Rcpp=%s\n",
                f, format(v_r), format(v_c)))
  }
  cat("--- end diagnostic ---\n")
}

## ---- Equivalence helper ----------------------------------------------------
## Does the actual equivalence assertion + diagnostic.
.check_engine_equivalence <- function(label, X, y, X_test, kernel, sym,
                                       block_k4) {
  fit_r <- suppressWarnings(psvr(
    X, y, loss = "mape", sym = sym, kernel = kernel,
    C = 10, eps = 5,
    engine = "r", block_k4_enabled = block_k4
  ))
  fit_rcpp <- suppressWarnings(psvr(
    X, y, loss = "mape", sym = sym, kernel = kernel,
    C = 10, eps = 5,
    engine = "rcpp", block_k4_enabled = block_k4
  ))
  preds_r    <- predict(fit_r,    X_test)
  preds_rcpp <- predict(fit_rcpp, X_test)
  if (!identical(preds_rcpp, preds_r)) {
    .diagnose_engine_diff(fit_r, fit_rcpp, preds_r, preds_rcpp,
                          label = label)
  }
  # Predictions must be bit-identical.
  expect_identical(preds_rcpp, preds_r,
                   label = sprintf("[%s] predictions", label))
  # Top-level fit fields.
  expect_identical(fit_rcpp$alpha,      fit_r$alpha,
                   label = sprintf("[%s] alpha",      label))
  expect_identical(fit_rcpp$alpha_star, fit_r$alpha_star,
                   label = sprintf("[%s] alpha_star", label))
  expect_identical(fit_rcpp$b,          fit_r$b,
                   label = sprintf("[%s] b",          label))
  # solver_meta fields.
  m_r <- fit_r$solver_meta
  m_c <- fit_rcpp$solver_meta
  expect_identical(m_c$iters,                       m_r$iters,
                   label = sprintf("[%s] iters",                  label))
  expect_identical(m_c$converged,                   m_r$converged,
                   label = sprintf("[%s] converged",              label))
  expect_identical(m_c$joint_updates,               m_r$joint_updates,
                   label = sprintf("[%s] joint_updates",          label))
  expect_identical(m_c$k2_fallbacks,                m_r$k2_fallbacks,
                   label = sprintf("[%s] k2_fallbacks",           label))
  expect_identical(m_c$decoupling_rate,             m_r$decoupling_rate,
                   label = sprintf("[%s] decoupling_rate",        label))
  expect_identical(m_c$early_phase_decoupling_rate, m_r$early_phase_decoupling_rate,
                   label = sprintf("[%s] early_phase_decoupling_rate", label))
  expect_identical(m_c$late_phase_decoupling_rate,  m_r$late_phase_decoupling_rate,
                   label = sprintf("[%s] late_phase_decoupling_rate",  label))
  invisible(NULL)
}

## ---- 16-config matrix: Models × Kernels × block_k4 ------------------------
## Pre-existing pathology note (paper TODO #5): linear/polynomial kernels
## with MAPE-SVR hit max_iter without converging in some regimes. Both
## engines stall at the same trajectory state, so equivalence still
## holds — we just observe `iter == max_iter && !converged` identically
## under both engines.

KERNELS_EQ <- list(
  rbf      = make_kernel("rbf",        sigma = 1),
  linear   = make_kernel("linear"),
  poly_d2  = make_kernel("polynomial", degree = 2L, coef0 = 1),
  poly_d3  = make_kernel("polynomial", degree = 3L, coef0 = 1)
)

for (model_label in c("Model 1 MAPE", "Model 2 MAPE-sym")) {
  sym_val <- if (model_label == "Model 2 MAPE-sym") 1L else NULL
  for (k_name in names(KERNELS_EQ)) {
    for (bk4 in c(FALSE, TRUE)) {
      label <- sprintf("%s / %s / bk4=%s", model_label, k_name, bk4)
      test_that(sprintf("engine equivalence: %s", label), {
        fx <- make_eq_fixture()
        .check_engine_equivalence(label, fx$X, fx$y, fx$X_test,
                                  kernel = KERNELS_EQ[[k_name]],
                                  sym    = sym_val,
                                  block_k4 = bk4)
      })
    }
  }
}

## ---- Sanity: schema invariants under both engines --------------------------
## Even when results are bit-equal, confirm the FitResult schema returned
## by Rcpp matches the R wrapper's downstream expectations.

test_that("engine='rcpp' returns the full solver_meta schema", {
  fx <- make_eq_fixture()
  K  <- make_kernel("rbf", sigma = 1)
  fit <- suppressWarnings(psvr(fx$X, fx$y, loss = "mape", kernel = K,
                                C = 10, eps = 5,
                                engine = "rcpp", block_k4_enabled = TRUE))
  meta <- fit$solver_meta
  for (f in c("backend", "iters", "converged",
              "joint_updates", "k2_fallbacks", "decoupling_rate",
              "early_phase_decoupling_rate", "late_phase_decoupling_rate")) {
    expect_true(f %in% names(meta),
                info = sprintf("solver_meta missing field: %s", f))
  }
  # decoupling rates are numeric (not NA on a converging fit with joint updates).
  expect_true(is.numeric(meta$decoupling_rate))
  expect_gte(meta$decoupling_rate, 0)
  expect_lte(meta$decoupling_rate, 1)
})

## ---- Sanity: engine = "r" + block_k4_enabled = FALSE matches the
##              _snaps/block-k4.md F4 reproducibility gate.
## This is redundant with test-block-k4.R's first two tests under the new
## default engine = "rcpp" — but here we explicitly check engine = "r"
## still produces the F4 baseline. Provides regression coverage if the
## dispatcher ever bypasses the .smo_solve_r() path incorrectly.

test_that("engine='r' preserves the F4 baseline (snapshot match)", {
  fx <- make_eq_fixture()
  K  <- make_kernel("rbf", sigma = 1)
  fit <- psvr(fx$X, fx$y, loss = "mape", kernel = K, C = 10, eps = 5,
              engine = "r", block_k4_enabled = FALSE)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})
