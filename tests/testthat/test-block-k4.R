## F7 — Block-k=4 SMO with descent-guaranteed decoupling.
##
## Tests live here (rather than in test-smo-solve.R) because they require
## both `block_k4_enabled = FALSE` (F4 reproduction) and the F7 default
## branch, and the bit-identical gate (test #1 / #2) is the hard guarantee
## that backward-compat reproducibility is preserved.

set.seed(2026)

# Standard N=50 / p=5 / rlnorm(0.5) fixture, matching test-bit-identical.R
# and test-psvr-direct.R, so the F4-baseline snapshot here is comparable to
# the existing golden snapshots.
make_f7_fixture <- function() {
  set.seed(2026)
  X      <- matrix(stats::rnorm(50 * 5), 50, 5)
  y      <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
  X_test <- matrix(stats::rnorm(20 * 5), 20, 5)
  list(X = X, y = y, X_test = X_test)
}

K_rbf <- make_kernel("rbf", sigma = 1)

# ---- 1. F4 reproducibility gate — Model 1 (MAPE, non-symmetric) -----------
# block_k4_enabled = FALSE must reproduce F4 behavior bit-identically. The
# stored snapshot equals the pre-F7 golden snapshot (from
# tests/testthat/_snaps/bit-identical.md, the "Model 1 mape_svr (RBF) —
# direct golden" entry); the FALSE-path is the backward-compat hard gate.

test_that("block_k4_enabled = FALSE: Model 1 MAPE reproduces F4 baseline", {
  # Golden baseline recorded on one x86_64 toolchain; see helper-fp-tiers.R.
  skip_on_cran()
  fx  <- make_f7_fixture()
  fit <- psvr_mape(fx$X, fx$y, kernel = K_rbf,
                   C = 10, eps = 5, block_k4_enabled = FALSE)

  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)

  meta <- fit$block_k4
  expect_equal(meta$joint_updates,    0L)
  expect_equal(meta$k2_fallbacks,     0L)
  expect_true(is.na(meta$decoupling_rate))
  expect_true(isTRUE(fit$converged))
})

test_that("block_k4_enabled = FALSE: Model 2 MAPE-sym reproduces F4 baseline", {
  # Golden baseline recorded on one x86_64 toolchain; see helper-fp-tiers.R.
  skip_on_cran()
  fx  <- make_f7_fixture()
  fit <- psvr_mape(fx$X, fx$y, sym_type = "even", kernel = K_rbf,
                   C = 10, eps = 5, block_k4_enabled = FALSE)

  preds <- predict(fit, fx$X_test)
  expect_snapshot_value(preds, style = "serialize", tolerance = 1e-10)

  meta <- fit$block_k4
  expect_equal(meta$joint_updates, 0L)
  expect_equal(meta$k2_fallbacks,  0L)
  expect_true(isTRUE(fit$converged))
})

# ---- 2. block_k4_enabled = TRUE converges to KKT-equivalent endpoint -------
# The block-k=4 trajectory reaches a different intermediate state than F4
# but must converge to a KKT-optimal fit. Predictions need not be bit-equal
# (different trajectory hits a different but KKT-equivalent endpoint); we
# allow ~tol*max(y) * 100 of slack to cover floating-point divergence at
# the per-pair tolerance floor.

test_that("block_k4_enabled = TRUE: converges and predictions track FALSE", {
  fx <- make_f7_fixture()

  fit_off <- psvr_mape(fx$X, fx$y, kernel = K_rbf,
                       C = 10, eps = 5, block_k4_enabled = FALSE)
  fit_on  <- psvr_mape(fx$X, fx$y, kernel = K_rbf,
                       C = 10, eps = 5, block_k4_enabled = TRUE)

  expect_true(isTRUE(fit_on$converged))
  expect_true(isTRUE(fit_off$converged))

  preds_off <- predict(fit_off, fx$X_test)
  preds_on  <- predict(fit_on,  fx$X_test)

  # tol_pair convergence floor at psvr_mape() default tol = 1e-5: ~ 1e-5 * max(y)
  # ~ 2.5e-5 in tau-space. Prediction-space drift can be 30x higher due to
  # ill-conditioning of the dual->prediction map (beta scaled by 100*C/y);
  # 1e-2 * mean(y) is comfortable for any KKT-equivalent endpoint pair.
  expect_lt(max(abs(preds_on - preds_off)), 1e-2 * mean(fx$y))
})

# ---- 3. Decoupling rate is reported and in [0, 1] -------------------------

test_that("decoupling_rate is reported in [0, 1] when block_k4_enabled = TRUE", {
  fx  <- make_f7_fixture()
  fit <- psvr_mape(fx$X, fx$y, kernel = K_rbf,
                   C = 10, eps = 5, block_k4_enabled = TRUE)

  meta <- fit$block_k4
  expect_gte(meta$joint_updates, 0L)
  expect_gte(meta$k2_fallbacks,  0L)

  if (!is.na(meta$decoupling_rate)) {
    expect_gte(meta$decoupling_rate, 0)
    expect_lte(meta$decoupling_rate, 1)
  }

  # joint + k2 == iters only if no iteration hit an early break before the
  # joint-decision section (unshrink-next, convergence break, etc.). The
  # weaker invariant is the safe one.
  expect_lte(meta$joint_updates + meta$k2_fallbacks, fit$iterations)
})

# ---- 4. Tiny fixture: |I_up| or |I_down| may be {1} or empty ---------------
# Edge case: 5-sample fit. The block-k=4 path must not crash when there is
# no candidate i_2 (because I_up \ {i_1} is empty) nor j_2 (candidate pool
# after sample-disjointness + tol filter is empty). All iterations should
# fall back to k=2 (or hit the bias-recovery path).

test_that("block-k=4 handles tiny fixtures without crashing", {
  set.seed(2026)
  N_tiny <- 5L
  X_t    <- matrix(rnorm(N_tiny * 2), N_tiny, 2)
  y_t    <- abs(rnorm(N_tiny)) + 1
  K_t    <- make_kernel("rbf", sigma = 1)

  expect_no_error({
    fit <- psvr_mape(X_t, y_t, kernel = K_t,
                     C = 10, eps = 5, block_k4_enabled = TRUE)
  })

  preds <- predict(fit, X_t)
  expect_true(all(is.finite(preds)))
  expect_true(all(preds > 0))
})

# ---- 5. Joint update preserves the equality constraint --------------------

test_that("joint update preserves sum(alpha - alpha_star) = 0", {
  fx  <- make_f7_fixture()
  fit <- psvr_mape(fx$X, fx$y, kernel = K_rbf,
                   C = 10, eps = 5, block_k4_enabled = TRUE)

  # Box-scaled tolerance, matching test-smo-solve.R test #8.
  expect_lt(abs(sum(fit$alpha - fit$alpha_star)),
            1e-8 * max(100 * 10 / fx$y))
})

# ---- 6. alpha_couple influences the decoupling rate -----------------------
# alpha_couple = 0 disables the coupling penalty (raw gain scoring; equivalent
# to greedy WSS3 on pair 2). alpha_couple = 1 fully suppresses max-correlation
# candidates. The trajectory should differ.

test_that("alpha_couple parameter influences decoupling trajectory", {
  fx <- make_f7_fixture()

  fit0 <- psvr_mape(fx$X, fx$y, kernel = K_rbf, C = 10, eps = 5,
                    block_k4_enabled = TRUE, alpha_couple = 0)
  fit1 <- psvr_mape(fx$X, fx$y, kernel = K_rbf, C = 10, eps = 5,
                    block_k4_enabled = TRUE, alpha_couple = 1)

  # If decoupling triggers in EITHER run, the trajectories should differ.
  # On a converged fit, identical decoupling rates AND identical iter counts
  # would suggest the coupling penalty had no effect, which would be a
  # regression in the D2 selection logic.
  if (fit0$block_k4$joint_updates > 0L ||
      fit1$block_k4$joint_updates > 0L) {
    differ <- !identical(fit0$block_k4$decoupling_rate,
                         fit1$block_k4$decoupling_rate) ||
              !identical(fit0$iterations,
                         fit1$iterations)
    expect_true(differ)
  }
})

# ---- 7. Default-collapse on near-homogeneous targets ----------------------
# rho_y ~ 1.05: T3+T8 (F4) reduce to defaults, T7 should remain functional
# and converge. Coverage check.

test_that("block-k=4 converges on near-homogeneous targets (rho_y ~ 1.05)", {
  set.seed(2026)
  N_h <- 50L
  X_h <- matrix(rnorm(N_h * 5), N_h, 5)
  y_h <- rep(2.0, N_h) + rnorm(N_h, sd = 0.05)
  K_h <- make_kernel("rbf", sigma = 1)

  fit <- psvr_mape(X_h, y_h, kernel = K_h, C = 10, eps = 5,
                   block_k4_enabled = TRUE)

  preds <- predict(fit, X_h)
  expect_true(all(is.finite(preds)))
  expect_true(all(preds > 0))
  rel_err <- abs(preds - y_h) / y_h
  expect_lt(median(rel_err), 0.10)
})

# ---- 8. block_k4 telemetry schema completeness ----------------------------
# All five F7 telemetry fields are present on the fit object, with the right
# types, for both block_k4_enabled = TRUE and = FALSE.

test_that("the fit object exposes the full F7 telemetry schema", {
  fx  <- make_f7_fixture()

  for (enabled in c(TRUE, FALSE)) {
    fit  <- psvr_mape(fx$X, fx$y, kernel = K_rbf,
                      C = 10, eps = 5, block_k4_enabled = enabled)
    meta <- fit$block_k4

    expect_true(is.integer(meta$joint_updates))
    expect_true(is.integer(meta$k2_fallbacks))
    expect_true(is.numeric(meta$decoupling_rate))            # may be NA_real_
    expect_true(is.numeric(meta$early_phase_decoupling_rate))
    expect_true(is.numeric(meta$late_phase_decoupling_rate))

    if (!enabled) {
      expect_equal(meta$joint_updates, 0L)
      expect_equal(meta$k2_fallbacks,  0L)
      expect_true(is.na(meta$decoupling_rate))
      expect_true(is.na(meta$early_phase_decoupling_rate))
      expect_true(is.na(meta$late_phase_decoupling_rate))
    }
  }
})

# ---- 9. block_k4_enabled does not exist on the LS-SVR entry point ---------
# This block used to assert that psvr(loss = "rmspe", block_k4_enabled = ...)
# ACCEPTED the argument and ignored it, leaving the telemetry at 0L / NA. That
# was a property of the one-function API: `block_k4_enabled` had to be a formal
# because MAPE needed it, so the RMSPE path had to tolerate it.
#
# psvr_rmspe() has no such formal, so the argument is now rejected rather than
# tolerated -- which is the stronger contract and the reason the split was
# worth doing. LS-SVR uses base::solve(), not SMO; there was never anything for
# block-k=4 to do here.

test_that("psvr_rmspe() has no block_k4_enabled argument", {
  fx <- make_f7_fixture()
  expect_error(
    psvr_rmspe(fx$X, fx$y, kernel = K_rbf, gamma = 100,
               block_k4_enabled = TRUE),
    class = "rlib_error_dots_nonempty"
  )
})
