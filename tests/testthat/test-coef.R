## coef() contract across BOTH entry points.
##
## coef.psvr_fit had no test at all before 0.0.2.9011 -- the only one of the
## five coef methods that was already correct was the untested one. The four
## legacy methods were tested, and two of them pinned the wrong name.
##
## The defect this file exists to prevent: `coef(fit)$alpha` meant the length-N
## dual via psvr() and the length-n_sv pruned beta via the legacy classes that
## parsnip::extract_fit_engine() returns. Same generic, same field name, two
## different vectors, no warning. The congruence block below is the regression
## test -- it fails if the two entry points ever disagree again.
##
## This file is deliberately separate from test-psvr-fit-shape.R, which pins the
## psvr_fit *object* shape and is expected to go away if psvr() is split.

set.seed(77)
X_tr <- matrix(runif(60, 0.5, 3), 20, 3)
y_tr <- 2 + X_tr[, 1]^2 + abs(rnorm(20, 0, 0.1))
K    <- make_kernel("rbf", sigma = 1)

# ── coef.psvr_fit — MAPE ─────────────────────────────────────────────────────

test_that("coef.psvr_fit on a MAPE fit exposes alpha, alpha_star, beta separately", {
  fit <- psvr(X_tr, y_tr, loss = "mape", kernel = K, C = 10, eps = 5)
  co  <- coef(fit)
  expect_named(co, c("alpha", "alpha_star", "beta", "b", "support_data"))
  expect_identical(co$alpha,        fit$alpha)
  expect_identical(co$alpha_star,   fit$alpha_star)
  expect_identical(co$beta,         fit$beta)
  expect_identical(co$b,            fit$b)
  expect_identical(co$support_data, fit$support_data)
  # The two lengths that the pre-0.0.2.9011 single `alpha` name conflated.
  expect_length(co$alpha,      nrow(X_tr))
  expect_length(co$alpha_star, nrow(X_tr))
  expect_length(co$beta,       fit$n_sv)
  # beta is the pruned alpha - alpha_star, not an independent quantity.
  full <- co$alpha - co$alpha_star
  expect_equal(full[abs(full) > 1e-5], co$beta, tolerance = 1e-10)
})

# ── coef.psvr_fit — RMSPE ────────────────────────────────────────────────────

test_that("coef.psvr_fit on an RMSPE fit carries alpha_star and beta as NULL", {
  fit <- psvr(X_tr, y_tr, loss = "rmspe", kernel = K, gamma = 100)
  co  <- coef(fit)
  # One class serves both families, so the slots are present-and-NULL rather
  # than absent. That is psvr_fit-specific; the legacy classes simply omit them.
  expect_named(co, c("alpha", "alpha_star", "beta", "b", "support_data"))
  expect_null(co$alpha_star)
  expect_null(co$beta)
  expect_identical(co$alpha,        fit$alpha)
  expect_identical(co$b,            fit$b)
  expect_identical(co$support_data, fit$support_data)
  expect_length(co$alpha, nrow(X_tr))
  # LS-SVR does no pruning: support_data is every training row, not a subset.
  expect_identical(nrow(co$support_data), nrow(X_tr))
})

test_that("coef.psvr_fit is unaffected by sym", {
  for (a in list(NULL, 1L, -1L)) {
    fit <- psvr(X_tr, y_tr, loss = "rmspe", sym = a, kernel = K, gamma = 100)
    expect_named(coef(fit),
                 c("alpha", "alpha_star", "beta", "b", "support_data"),
                 info = paste("sym =", if (is.null(a)) "NULL" else a))
  }
})

# ── cross-entry-point congruence — the regression test ───────────────────────

test_that("coef() agrees across psvr() and the legacy MAPE classes", {
  # psvr:::.fit_mape() is what the parsnip engine fit wrappers call, so its
  # return is what extract_fit_engine() hands back.
  new <- coef(psvr(X_tr, y_tr, loss = "mape", kernel = K, C = 10, eps = 5))
  old <- coef(psvr:::.fit_mape(X_tr, y_tr, kernel = K, C = 10, eps = 5))
  expect_identical(names(new), names(old))
  expect_identical(new, old)
})

test_that("coef() agrees across psvr() and the legacy symmetric MAPE classes", {
  new <- coef(psvr(X_tr, y_tr, loss = "mape", sym = 1L,
                   kernel = K, C = 10, eps = 5))
  old <- coef(psvr:::.fit_mape_sym(X_tr, y_tr, kernel = K, C = 10, eps = 5,
                                   a = 1L))
  expect_identical(names(new), names(old))
  expect_identical(new, old)
})

test_that("coef() values agree across psvr() and the legacy LS-SVR classes", {
  # Names differ by design here: psvr_fit materialises alpha_star and beta as
  # NULL because one class serves both families, while the legacy LS-SVR classes
  # omit them. `$` returns NULL either way, so every accessor still agrees.
  for (a in list(NULL, 1L)) {
    new <- coef(psvr(X_tr, y_tr, loss = "rmspe", sym = a,
                     kernel = K, gamma = 100))
    old <- if (is.null(a)) {
      coef(psvr:::.fit_rmspe(X_tr, y_tr, kernel = K, gamma = 100))
    } else {
      coef(psvr:::.fit_rmspe_sym(X_tr, y_tr, kernel = K, gamma = 100, a = a))
    }
    lbl <- paste("sym =", if (is.null(a)) "NULL" else a)
    for (nm in c("alpha", "alpha_star", "beta", "b", "support_data")) {
      expect_identical(new[[nm]], old[[nm]], info = paste(lbl, "/", nm))
    }
  }
})
