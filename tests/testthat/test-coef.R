## coef() contract on the four fit classes.
##
## The defect this file was written for (0.0.2.9011): `coef(fit)$alpha` meant
## the length-N dual via psvr() and the length-n_sv pruned beta via the classes
## parsnip::extract_fit_engine() returns. Same generic, same component name,
## two different vectors, no warning.
##
## API-redesign stage 5 supersedes the second entry point rather than the
## disagreement: psvr_mape() and psvr_rmspe() return the same four classes the
## parsnip wrappers do, so there is now exactly one coef() implementation per
## class and nothing left to diverge. The congruence blocks that used to close
## this file went with it -- see the note at the bottom for why keeping them
## would have been theatre.
##
## What survives is the contract itself: which components each family returns,
## what they mean, and what length they are.

set.seed(77)
X_tr <- matrix(runif(60, 0.5, 3), 20, 3)
y_tr <- 2 + X_tr[, 1]^2 + abs(rnorm(20, 0, 0.1))
K    <- make_kernel("rbf", sigma = 1)

# ── coef.psvr_fit — MAPE ─────────────────────────────────────────────────────

test_that("coef.psvr_mape exposes alpha, alpha_star, beta separately", {
  fit <- psvr_mape(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  co  <- coef(fit)
  expect_named(co, c("alpha", "alpha_star", "beta", "b", "support_data"))
  expect_identical(co$alpha,        fit$alpha)
  expect_identical(co$alpha_star,   fit$alpha_star)
  expect_identical(co$beta,         fit$beta)
  expect_identical(co$b,            fit$b)
  # coef() renamed the component to `support_data` in 0.0.2.9011; the FIELD on
  # the fit object is still `X_sv`. Comparing against `fit$support_data` would
  # compare a matrix with NULL and pass nothing useful.
  expect_identical(co$support_data, fit$X_sv)
  # The two lengths that the pre-0.0.2.9011 single `alpha` name conflated.
  expect_length(co$alpha,      nrow(X_tr))
  expect_length(co$alpha_star, nrow(X_tr))
  expect_length(co$beta,       length(fit$beta))
  # beta is the pruned alpha - alpha_star, not an independent quantity.
  full <- co$alpha - co$alpha_star
  expect_equal(full[abs(full) > 1e-5], co$beta, tolerance = 1e-10)
})

# ── coef.psvr_fit — RMSPE ────────────────────────────────────────────────────

test_that("coef.psvr_rmspe returns three components, not five", {
  fit <- psvr_rmspe(X_tr, y_tr, kernel = K, gamma = 100)
  co  <- coef(fit)
  # LS-SVR has no alpha_star and no pruned beta, and the absent components are
  # NOT materialised as NULL. So names(coef(fit)) depends on the model family.
  # That is a decision, not an oversight: each class is family-specific, and
  # inventing empty slots to make the two agree would add structure with
  # nothing to inherit it from. `$` still yields NULL for both, so every
  # accessor agrees with the MAPE classes even though names() does not.
  expect_named(co, c("alpha", "b", "support_data"))
  expect_null(co$alpha_star)
  expect_null(co$beta)
  expect_identical(co$alpha,        fit$alpha)
  expect_identical(co$b,            fit$b)
  # As above: the field is `X_train` on an LS-SVR fit. LS-SVR does no pruning,
  # which is why `X_sv` was the wrong name for it and 0.0.2.9011 renamed the
  # coef() component to `support_data`.
  expect_identical(co$support_data, fit$X_train)
  expect_length(co$alpha, nrow(X_tr))
  # LS-SVR does no pruning: support_data is every training row, not a subset.
  expect_identical(nrow(co$support_data), nrow(X_tr))
})

test_that("coef on the LS-SVR classes is unaffected by symmetry", {
  for (s in c("none", "even", "odd")) {
    fit <- psvr_rmspe(X_tr, y_tr, sym_type = s, kernel = K, gamma = 100)
    expect_named(coef(fit), c("alpha", "b", "support_data"),
                 info = paste("sym_type =", s))
  }
})

# ── the cross-entry-point congruence blocks, and why they are gone ───────────
#
# Three blocks lived here. They compared coef(psvr(...)) against coef() on the
# corresponding object from psvr:::.fit_*(), and existed to catch a repeat of
# the 0.0.2.9011 defect: one generic returning two different vectors under the
# name `alpha` depending on which entry point produced the fit.
#
# The API split DISSOLVES that defect class rather than relocating it.
# psvr_mape() IS .fit_mape() -- it validates, maps sym_type to `a`, and returns
# the fitter's object unwrapped -- so the comparison became
# coef(x) vs coef(x): a tautology that cannot fail, and therefore cannot
# guard anything. Keeping it would have been theatre.
#
# The contract those three stood in for is now asserted directly by blocks 1-3
# above, which are the only coef() implementations left. There is no second
# entry point for them to disagree with.
