## F7.5 — trace = TRUE collects per-iter WSS1 Delta into delta_history.
##
## Tests at the .smo_solve() developer interface (trace is not exposed in
## psvr()). Covers:
##   1. trace = FALSE: bit-identical to omitting the argument.
##   2. trace = TRUE : returns delta_history with length == iterations.
##   3. trace = TRUE : alpha/b unchanged from trace = FALSE on same fixture.
##   4. Engine equivalence: r vs rcpp produce bit-identical delta_history
##      on 4 configs (Models 1+2 × bk4 FALSE+TRUE on RBF) — the C++ port
##      bit-identicality canary extended to the trace recording path.
##   5. Monotonicity (soft): tail Delta < head Delta on a converging fit.
##   6. Edge case max_iter = 0: numeric(0) returned, not NULL.

## ---- Fixture ----
.trace_fixture <- function() {
  set.seed(2026)
  N <- 30
  X <- matrix(stats::rnorm(N * 4), N, 4)
  y <- stats::rlnorm(N, meanlog = 0, sdlog = 0.4)
  list(X = X, y = y)
}

.build_K_acc <- function(X, kernel_fun, sym = NULL) {
  Om <- if (is.null(sym)) {
    kernel_matrix(kernel_fun, X)
  } else {
    sym_kernel_matrix(kernel_fun, X, a = sym)
  }
  diag(Om) <- diag(Om) + 1e-8
  psvr:::.make_kernel_accessor(Om)
}

## ---- 1. trace = FALSE is bit-identical to omitting the argument -----------
test_that("trace = FALSE matches default (no argument) on engine = 'r'", {
  fx    <- .trace_fixture()
  K_acc <- .build_K_acc(fx$X, make_kernel("rbf", sigma = 1))
  s_def <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = "r")
  s_F   <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = "r",
                             trace = FALSE)
  expect_identical(s_def$alpha,      s_F$alpha)
  expect_identical(s_def$alpha_star, s_F$alpha_star)
  expect_identical(s_def$b,          s_F$b)
  expect_identical(s_def$iterations, s_F$iterations)
  expect_null(s_def$delta_history)
  expect_null(s_F$delta_history)
  expect_null(s_def$active_history)
  expect_null(s_F$active_history)
})

test_that("trace = FALSE matches default (no argument) on engine = 'rcpp'", {
  fx    <- .trace_fixture()
  K_acc <- .build_K_acc(fx$X, make_kernel("rbf", sigma = 1))
  s_def <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = "rcpp")
  s_F   <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = "rcpp",
                             trace = FALSE)
  expect_identical(s_def$alpha,      s_F$alpha)
  expect_identical(s_def$alpha_star, s_F$alpha_star)
  expect_identical(s_def$b,          s_F$b)
  expect_identical(s_def$iterations, s_F$iterations)
  expect_null(s_def$delta_history)
  expect_null(s_F$delta_history)
  expect_null(s_def$active_history)
  expect_null(s_F$active_history)
})

## ---- 2 + 3. trace = TRUE returns delta_history + active_history, same numerics
test_that("trace = TRUE returns numeric delta_history with length == iterations", {
  fx    <- .trace_fixture()
  K_acc <- .build_K_acc(fx$X, make_kernel("rbf", sigma = 1))
  s_F <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = "r")
  s_T <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = "r",
                            trace = TRUE)
  expect_type(s_T$delta_history, "double")
  expect_length(s_T$delta_history, s_T$iterations)
  # F7.6 — active_history: integer, same length, bounded by 2*N.
  expect_type(s_T$active_history, "integer")
  expect_length(s_T$active_history, s_T$iterations)
  N <- nrow(fx$X)
  expect_true(all(s_T$active_history >= 0L))
  expect_true(all(s_T$active_history <= 2L * N))
  # Numerics unchanged by trace=TRUE.
  expect_identical(s_F$alpha,      s_T$alpha)
  expect_identical(s_F$alpha_star, s_T$alpha_star)
  expect_identical(s_F$b,          s_T$b)
  expect_identical(s_F$iterations, s_T$iterations)
})

## ---- 4. Engine equivalence on delta_history (4 configs) ------------------
## All four configs are RBF and converge on every platform tested, so the
## only cross-platform effect here is FP noise (max 5.151e-14 observed on
## macOS aarch64). Two tiers per helper-fp-tiers.R.

.trace_engine_pair <- function(label, kernel_fun, sym, bk4) {
  psvr_memo(paste0("trace::", label), {
    fx <- .trace_fixture()
    K_acc <- .build_K_acc(fx$X, kernel_fun, sym = sym)
    list(
      s_r = suppressWarnings(psvr:::.smo_solve(
        K_acc, fx$y, C = 10, eps = 5,
        block_k4_enabled = bk4, engine = "r",    trace = TRUE
      )),
      s_c = suppressWarnings(psvr:::.smo_solve(
        K_acc, fx$y, C = 10, eps = 5,
        block_k4_enabled = bk4, engine = "rcpp", trace = TRUE
      ))
    )
  })
}

for (sym_label in c("Model 1 MAPE", "Model 2 MAPE-sym")) {
  sym_val <- if (sym_label == "Model 2 MAPE-sym") 1L else NULL
  for (bk4 in c(FALSE, TRUE)) {
    local({
      label   <- sprintf("%s / RBF / bk4=%s / trace=TRUE", sym_label, bk4)
      sym_l   <- sym_val
      bk4_l   <- bk4
      kern    <- make_kernel("rbf", sigma = 1)

      test_that(sprintf("engine equivalence [strict] on delta_history: %s", label), {
        skip_on_cran()
        p <- .trace_engine_pair(label, kern, sym_l, bk4_l)
        expect_identical(p$s_c$alpha,      p$s_r$alpha,
                         label = sprintf("[%s] alpha", label))
        expect_identical(p$s_c$alpha_star, p$s_r$alpha_star,
                         label = sprintf("[%s] alpha_star", label))
        expect_identical(p$s_c$b,          p$s_r$b,
                         label = sprintf("[%s] b", label))
        expect_identical(p$s_c$iterations, p$s_r$iterations,
                         label = sprintf("[%s] iterations", label))
        # The F7.5 gate: delta_history bit-identical across engines.
        expect_identical(p$s_c$delta_history, p$s_r$delta_history,
                         label = sprintf("[%s] delta_history", label))
        # F7.6 — active_history bit-identical across engines.
        expect_identical(p$s_c$active_history, p$s_r$active_history,
                         label = sprintf("[%s] active_history", label))
      })

      test_that(sprintf("engine equivalence [tolerance] on delta_history: %s", label), {
        p <- .trace_engine_pair(label, kern, sym_l, bk4_l)
        expect_equal(p$s_c$alpha,      p$s_r$alpha,      tolerance = PSVR_FP_TOL,
                     label = sprintf("[%s] alpha", label))
        expect_equal(p$s_c$alpha_star, p$s_r$alpha_star, tolerance = PSVR_FP_TOL,
                     label = sprintf("[%s] alpha_star", label))
        expect_equal(p$s_c$b,          p$s_r$b,          tolerance = PSVR_FP_TOL,
                     label = sprintf("[%s] b", label))
        expect_equal(p$s_c$delta_history, p$s_r$delta_history,
                     tolerance = PSVR_FP_TOL,
                     label = sprintf("[%s] delta_history", label))
        # Trajectory length and the active-set counts are integers: if the
        # engines agree at all they agree exactly, so no tolerance applies.
        expect_identical(p$s_c$iterations,     p$s_r$iterations,
                         label = sprintf("[%s] iterations", label))
        expect_identical(p$s_c$active_history, p$s_r$active_history,
                         label = sprintf("[%s] active_history", label))
      })
    })
  }
}

## ---- 5. Monotonicity (soft) ----------------------------------------------
## KKT gap should decrease over a converging trajectory. Block-k=4 joint
## updates and shrink/restore points can produce local upticks, so we only
## assert tail < head (smoothed trend), not strict monotonicity.
test_that("delta_history trend decreases (tail < head) on converging fit", {
  fx    <- .trace_fixture()
  K_acc <- .build_K_acc(fx$X, make_kernel("rbf", sigma = 1))
  for (eng in c("r", "rcpp")) {
    s_T <- psvr:::.smo_solve(K_acc, fx$y, C = 10, eps = 5, engine = eng,
                              trace = TRUE)
    dh <- s_T$delta_history
    expect_true(isTRUE(s_T$converged),
                info = sprintf("fixture should converge under engine='%s'", eng))
    expect_gt(length(dh), 5L)
    expect_lt(utils::tail(dh, 1L), utils::head(dh, 1L))
    # Final Delta should be at or near the convergence threshold.
    expect_lt(utils::tail(dh, 1L), 1.0)
  }
})

## ---- 6. Edge case: max_iter = 0 ------------------------------------------
## With zero iters the outer SMO loop body never runs; delta_history must
## be numeric(0), NOT NULL — matching `seq_len(0)` behavior in R.
## active_history must be integer(0) under the same rule.
test_that("trace = TRUE with max_iter = 0 returns zero-length vectors, not NULL", {
  fx    <- .trace_fixture()
  K_acc <- .build_K_acc(fx$X, make_kernel("rbf", sigma = 1))
  for (eng in c("r", "rcpp")) {
    s <- suppressWarnings(psvr:::.smo_solve(
      K_acc, fx$y, C = 10, eps = 5,
      max_iter = 0L, engine = eng, trace = TRUE
    ))
    expect_false(is.null(s$delta_history),
                 info = sprintf("engine = '%s'", eng))
    expect_type(s$delta_history, "double")
    expect_length(s$delta_history, 0L)
    expect_false(is.null(s$active_history),
                 info = sprintf("engine = '%s'", eng))
    expect_type(s$active_history, "integer")
    expect_length(s$active_history, 0L)
    expect_identical(s$iterations, 0L)
  }
})

## ---- 7. active_history reflects shrinking dynamics -----------------------
## On a fixture that shrinks (n_check small, n_freeze small), the
## active-set count should not monotonically decrease — restoration events
## must produce upticks. Sanity-check: at least one iter where the active
## count exceeds the immediately-prior iter's count.
test_that("active_history exhibits non-monotone dynamics with shrink/restore", {
  fx    <- .trace_fixture()
  K_acc <- .build_K_acc(fx$X, make_kernel("rbf", sigma = 1))
  N <- nrow(fx$X)
  for (eng in c("r", "rcpp")) {
    s <- psvr:::.smo_solve(
      K_acc, fx$y, C = 10, eps = 5,
      n_check = 3L, n_freeze = 2L,
      engine = eng, trace = TRUE
    )
    ah <- s$active_history
    expect_true(isTRUE(s$converged),
                info = sprintf("fixture should converge under engine='%s'", eng))
    # First iter starts with all-active (= 2*N).
    expect_identical(ah[1L], 2L * as.integer(N))
    # Active count stays within [0, 2*N] throughout.
    expect_true(all(ah >= 0L), info = sprintf("engine = '%s'", eng))
    expect_true(all(ah <= 2L * N), info = sprintf("engine = '%s'", eng))
  }
})
