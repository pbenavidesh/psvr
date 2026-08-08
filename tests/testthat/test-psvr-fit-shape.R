## Exact-shape pin for the CURRENT psvr_fit class.
##
## Semantics differ deliberately from the legacy shape tests
## (test-mape-svr.R et al.), which assert a SUBSET because they pin a
## deprecated shape where the contract is "nothing was removed". Here the
## class becomes public API at 0.1.0, so the contract is the stronger
## "tell me when the shape changes" -- exact names, exact order. A field
## added, removed, or reordered fails this test on purpose; if the change
## is intended, update the vector below in the same commit.

set.seed(9L)
X_tr <- matrix(runif(40, 0.5, 3), 20, 2)
y_tr <- 2 + X_tr[, 1]^2 + abs(rnorm(20, 0, 0.1))
K    <- make_kernel("rbf", sigma = 0.8)

# All four model configurations. NULL-valued components (e.g. `sym` on a
# non-symmetric fit) are preserved by list(), so the name vector is the
# same for all four -- that invariant is itself asserted below.
.fits <- function() {
  suppressWarnings(list(
    mape_std  = psvr(X_tr, y_tr, loss = "mape",  kernel = K, C = 1, eps = 0.1),
    mape_sym  = psvr(X_tr, y_tr, loss = "mape",  kernel = K, C = 1, eps = 0.1,
                     sym = 1L),
    rmspe_std = psvr(X_tr, y_tr, loss = "rmspe", kernel = K, gamma = 10),
    rmspe_sym = psvr(X_tr, y_tr, loss = "rmspe", kernel = K, gamma = 10,
                     sym = -1L)
  ))
}

PSVR_FIT_NAMES <- c(
  "loss", "sym", "kernel",
  "alpha", "alpha_star", "beta", "b",
  "support_data", "support_targets",
  "y_train", "fitted_values",
  "n_train", "n_sv", "p_train",
  "hyperparameters", "solver_meta"
)

PSVR_FIT_HYPER <- c("C", "eps", "gamma", "a")

PSVR_FIT_META <- c(
  "backend", "iters", "converged", "precondition_applied", "spectral",
  "joint_updates", "k2_fallbacks", "decoupling_rate",
  "early_phase_decoupling_rate", "late_phase_decoupling_rate"
)

test_that("psvr_fit has the exact documented shape in all four configurations", {
  for (nm in names(.fits())) {
    fit <- .fits()[[nm]]
    expect_s3_class(fit, "psvr_fit")
    expect_named(fit, PSVR_FIT_NAMES, info = nm)
    expect_length(fit, length(PSVR_FIT_NAMES))
  }
})

test_that("the nested hyperparameters and solver_meta shapes are exact too", {
  for (nm in names(.fits())) {
    fit <- .fits()[[nm]]
    expect_named(fit$hyperparameters, PSVR_FIT_HYPER, info = nm)
    expect_named(fit$solver_meta,     PSVR_FIT_META,  info = nm)
  }
})

test_that("all four configurations share one name vector", {
  # Guards the list()-preserves-NULL behaviour the pin above relies on: if a
  # NULL component were ever dropped instead, the shape would silently become
  # config-dependent.
  nms <- lapply(.fits(), names)
  expect_length(unique(nms), 1L)
})

test_that("family-dependent components are populated per the documented schema", {
  fits <- .fits()
  N <- nrow(X_tr)

  for (nm in c("mape_std", "mape_sym")) {
    f <- fits[[nm]]
    expect_length(f$alpha,      N)
    expect_length(f$alpha_star, N)
    expect_length(f$beta,       f$n_sv)
    expect_false(is.null(f$support_targets))
  }
  for (nm in c("rmspe_std", "rmspe_sym")) {
    f <- fits[[nm]]
    expect_length(f$alpha, N)
    expect_null(f$alpha_star)
    expect_null(f$beta)
    expect_null(f$support_targets)
    expect_identical(f$n_sv, N)
  }

  expect_null(fits$mape_std$sym)
  expect_null(fits$rmspe_std$sym)
  expect_identical(fits$mape_sym$sym,  1L)
  expect_identical(fits$rmspe_sym$sym, -1L)
})

test_that("the training-state components are length N in every configuration", {
  N <- nrow(X_tr)
  for (nm in names(.fits())) {
    fit <- .fits()[[nm]]
    expect_length(fit$y_train,       N)
    expect_length(fit$fitted_values, N)
    expect_identical(fit$n_train,    N)
    expect_true(is.numeric(fit$y_train),       info = nm)
    expect_true(is.numeric(fit$fitted_values), info = nm)
  }
})
