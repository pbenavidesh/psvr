## F7-C-full — engine="r" vs engine="rcpp" numerical equivalence canary.
##
## The strict bit-identicality gate for the C++ port: every fit configured
## (model × kernel × block_k4) must produce IDENTICAL doubles across the
## two engines. Predictions, solver telemetry, alpha, alpha_star, b — all
## bit-equal. The 16-config matrix below is the regression surface that
## keeps the engines in lockstep through v0.0.4.x deprecation and v0.1.0
## removal of the R reference.
##
## Diagnostic policy: a bare expect_identical() gives binary pass/fail
## with no FP-level context. On any failure we print max diff, first
## differing indices, side-by-side values at full precision, plus iteration
## counts and block-k=4 telemetry deltas. This feeds the escalation policy
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
##   * telemetry field deltas: alpha, alpha_star, b, joint_updates,
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
              fit_r$iterations, fit_rcpp$iterations,
              fit_rcpp$iterations - fit_r$iterations))
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
    v_r <- fit_r$block_k4[[f]]
    v_c <- fit_rcpp$block_k4[[f]]
    cat(sprintf("  block_k4$%-30s R=%s  Rcpp=%s\n",
                f, format(v_r), format(v_c)))
  }
  cat("--- end diagnostic ---\n")
}

## ---- Memoised fit pair -----------------------------------------------------
## One fit per engine per config, shared by the strict and tolerance tiers.
.eq_fits <- function(label, kernel, sym_type, block_k4) {
  psvr_memo(paste0("eqfit::", label), {
    fx <- make_eq_fixture()
    fit_r <- suppressWarnings(psvr_mape(
      fx$X, fx$y, sym_type = sym_type, kernel = kernel,
      C = 10, eps = 5,
      engine = "r", block_k4_enabled = block_k4
    ))
    fit_rcpp <- suppressWarnings(psvr_mape(
      fx$X, fx$y, sym_type = sym_type, kernel = kernel,
      C = 10, eps = 5,
      engine = "rcpp", block_k4_enabled = block_k4
    ))
    list(fit_r = fit_r, fit_rcpp = fit_rcpp,
         preds_r    = predict(fit_r,    fx$X_test),
         preds_rcpp = predict(fit_rcpp, fx$X_test))
  })
}

## ---- Strict tier -----------------------------------------------------------
## Bit-equality across engines, with the full diagnostic on failure.
.assert_engine_strict <- function(label, r) {
  if (!identical(r$preds_rcpp, r$preds_r)) {
    .diagnose_engine_diff(r$fit_r, r$fit_rcpp, r$preds_r, r$preds_rcpp,
                          label = label)
  }
  expect_identical(r$preds_rcpp, r$preds_r,
                   label = sprintf("[%s] predictions", label))
  expect_identical(r$fit_rcpp$alpha,      r$fit_r$alpha,
                   label = sprintf("[%s] alpha",      label))
  expect_identical(r$fit_rcpp$alpha_star, r$fit_r$alpha_star,
                   label = sprintf("[%s] alpha_star", label))
  expect_identical(r$fit_rcpp$b,          r$fit_r$b,
                   label = sprintf("[%s] b",          label))
  m_r <- r$fit_r
  m_c <- r$fit_rcpp
  expect_identical(m_c$iterations,                       m_r$iterations,
                   label = sprintf("[%s] iters",                  label))
  expect_identical(m_c$converged,                   m_r$converged,
                   label = sprintf("[%s] converged",              label))
  expect_identical(m_c$block_k4$joint_updates,               m_r$block_k4$joint_updates,
                   label = sprintf("[%s] joint_updates",          label))
  expect_identical(m_c$block_k4$k2_fallbacks,                m_r$block_k4$k2_fallbacks,
                   label = sprintf("[%s] k2_fallbacks",           label))
  expect_identical(m_c$block_k4$decoupling_rate,             m_r$block_k4$decoupling_rate,
                   label = sprintf("[%s] decoupling_rate",        label))
  expect_identical(m_c$block_k4$early_phase_decoupling_rate, m_r$block_k4$early_phase_decoupling_rate,
                   label = sprintf("[%s] early_phase_decoupling_rate", label))
  expect_identical(m_c$block_k4$late_phase_decoupling_rate,  m_r$block_k4$late_phase_decoupling_rate,
                   label = sprintf("[%s] late_phase_decoupling_rate",  label))
  invisible(NULL)
}

## ---- Tolerance tier --------------------------------------------------------
## Runs everywhere. Three regimes, in order:
##
##   1. Convergence-status pin (always asserted). The set of configs that
##      exhaust max_iter must equal PSVR_MAXITER_CONFIGS exactly. This is
##      what stops regime 2's skip from hiding a regression: a newly
##      non-converging config fails here instead of silently skipping, and a
##      config that starts converging also fails, forcing the pin to be
##      updated rather than the skip to widen.
##   2. Both engines at max_iter -> skip the value comparison. Two solvers
##      that both failed to converge agree on nothing meaningful.
##   3. Otherwise -> assert to PSVR_FP_TOL, or PSVR_MARGINAL_TOL for the
##      configs pinned in PSVR_MARGINAL_CONFIGS.
.assert_engine_tolerance <- function(label, r) {
  m_r <- r$fit_r
  m_c <- r$fit_rcpp

  # (1) Engines must always agree on whether they converged.
  expect_identical(m_c$converged, m_r$converged,
                   label = sprintf("[%s] converged", label))

  at_cap <- !isTRUE(m_r$converged) && !isTRUE(m_c$converged)
  expect_identical(
    at_cap, label %in% PSVR_MAXITER_CONFIGS,
    label = sprintf(
      paste0("[%s] max_iter status vs PSVR_MAXITER_CONFIGS pin ",
             "(at_cap=%s, pinned=%s; iters r=%s rcpp=%s). Update the pin in ",
             "helper-fp-tiers.R only after establishing why convergence changed"),
      label, at_cap, label %in% PSVR_MAXITER_CONFIGS,
      m_r$iterations, m_c$iterations)
  )

  # (2) Known non-convergence pathology: nothing meaningful to compare.
  if (at_cap) {
    skip(sprintf(
      paste0("[%s] both engines exhausted max_iter (%s) — documented ",
             "linear/polynomial MAPE-SVR non-convergence pathology ",
             "(CLAUDE.md 'Known issues', paper TODO #5). Value comparison ",
             "is not meaningful between two non-converged solutions; the ",
             "strict tier still gates this config on x86_64."),
      label, m_r$iterations))
  }

  # (3) Converged on both engines.
  tol <- if (label %in% PSVR_MARGINAL_CONFIGS) PSVR_MARGINAL_TOL else PSVR_FP_TOL
  expect_equal(r$preds_rcpp, r$preds_r, tolerance = tol,
               label = sprintf("[%s] predictions", label))
  expect_equal(r$fit_rcpp$b, r$fit_r$b, tolerance = tol,
               label = sprintf("[%s] b", label))

  if (label %in% PSVR_MARGINAL_CONFIGS) {
    # Trajectory length is platform-dependent here (6431 on x86_64 both
    # engines; 6435 R / 7795 Rcpp on aarch64), so iters and the duals are
    # not comparable. Predictions and b above are the meaningful check.
    return(invisible(NULL))
  }

  expect_equal(r$fit_rcpp$alpha,      r$fit_r$alpha,      tolerance = tol,
               label = sprintf("[%s] alpha", label))
  expect_equal(r$fit_rcpp$alpha_star, r$fit_r$alpha_star, tolerance = tol,
               label = sprintf("[%s] alpha_star", label))
  expect_identical(m_c$iterations, m_r$iterations,
                   label = sprintf("[%s] iters", label))
  expect_identical(m_c$block_k4$joint_updates, m_r$block_k4$joint_updates,
                   label = sprintf("[%s] joint_updates", label))
  expect_identical(m_c$block_k4$k2_fallbacks,  m_r$block_k4$k2_fallbacks,
                   label = sprintf("[%s] k2_fallbacks", label))
  invisible(NULL)
}

## ---- 16-config matrix: Models × Kernels × block_k4 ------------------------
## Pre-existing pathology note (paper TODO #5): linear/polynomial kernels
## with MAPE-SVR hit max_iter without converging in some regimes.
##
## On x86_64 both engines stall at the SAME trajectory state, so strict
## equivalence still holds there. On macOS aarch64 it does not: the two
## engines reach the cap at different non-converged points, diverging by up
## to 2.3e+00 in prediction space. See helper-fp-tiers.R for the tiering
## policy and PSVR_MAXITER_CONFIGS for the pinned set.

KERNELS_EQ <- list(
  rbf      = make_kernel("rbf",        sigma = 1),
  linear   = make_kernel("linear"),
  poly_d2  = make_kernel("polynomial", degree = 2L, coef0 = 1),
  poly_d3  = make_kernel("polynomial", degree = 3L, coef0 = 1)
)

for (model_label in c("Model 1 MAPE", "Model 2 MAPE-sym")) {
  sym_val <- if (model_label == "Model 2 MAPE-sym") "even" else "none"
  for (k_name in names(KERNELS_EQ)) {
    for (bk4 in c(FALSE, TRUE)) {
      local({
        label   <- sprintf("%s / %s / bk4=%s", model_label, k_name, bk4)
        kern    <- KERNELS_EQ[[k_name]]
        sym_l   <- sym_val
        bk4_l   <- bk4

        ## Strict tier — bit-equality. Intra-platform regression gate.
        test_that(sprintf("engine equivalence [strict]: %s", label), {
          skip_on_cran()
          r <- .eq_fits(label, kern, sym_l, bk4_l)
          .assert_engine_strict(label, r)
        })

        ## Tolerance tier — runs on every platform, including CRAN.
        test_that(sprintf("engine equivalence [tolerance]: %s", label), {
          r <- .eq_fits(label, kern, sym_l, bk4_l)
          .assert_engine_tolerance(label, r)
        })
      })
    }
  }
}

## ---- Sanity: schema invariants under both engines --------------------------
## Even when results are bit-equal, confirm the FitResult schema returned
## by Rcpp matches the R wrapper's downstream expectations.

test_that("engine='rcpp' returns the full solver telemetry schema", {
  fx <- make_eq_fixture()
  K  <- make_kernel("rbf", sigma = 1)
  fit <- suppressWarnings(psvr_mape(fx$X, fx$y, kernel = K,
                                    C = 10, eps = 5,
                                    engine = "rcpp",
                                    block_k4_enabled = TRUE))
  # `backend` is gone with psvr_fit: no fit class records which solver ran.
  # `iters` is now `iterations` at the top level, and the five block-k=4
  # counters live under `block_k4`.
  for (f in c("iterations", "converged", "block_k4")) {
    expect_true(f %in% names(fit),
                info = sprintf("fit object missing field: %s", f))
  }
  meta <- fit$block_k4
  for (f in c("joint_updates", "k2_fallbacks", "decoupling_rate",
              "early_phase_decoupling_rate", "late_phase_decoupling_rate")) {
    expect_true(f %in% names(meta),
                info = sprintf("block_k4 missing field: %s", f))
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
  # Golden baseline recorded on one x86_64 toolchain; see helper-fp-tiers.R.
  skip_on_cran()
  fx <- make_eq_fixture()
  K  <- make_kernel("rbf", sigma = 1)
  fit <- psvr_mape(fx$X, fx$y, kernel = K, C = 10, eps = 5,
                   engine = "r", block_k4_enabled = FALSE)
  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)
})
