## Smoke tests for all 12 parsnip model specs: fit + predict on a small
## synthetic dataset with strictly positive targets.

library(parsnip)

set.seed(42)
n    <- 20
X_tr <- matrix(runif(n * 2, 1, 5), ncol = 2,
               dimnames = list(NULL, c("V1", "V2")))
y_tr <- 2 + X_tr[, 1] + X_tr[, 2] + rnorm(n, sd = 0.1)   # y > 2, strictly positive
X_te <- matrix(c(2, 3, 3, 4), ncol = 2,
               dimnames = list(NULL, c("V1", "V2")))
df_te <- as.data.frame(X_te)
stopifnot(all(y_tr > 0))

smoke <- function(spec) {
  fit_obj  <- parsnip::fit_xy(spec, x = X_tr, y = y_tr)
  preds    <- predict(fit_obj, new_data = df_te)
  expect_s3_class(fit_obj, "model_fit")
  expect_equal(nrow(preds), nrow(X_te))
  expect_true(all(is.finite(preds$.pred)))
}

# ---- Model 1: epsilon-SVR with MAPE ----

test_that("psvr_mape_rbf smoke", {
  smoke(psvr_mape_rbf(cost = 10, svm_margin = 1, rbf_sigma = 1) |>
          set_engine("psvr"))
})

test_that("psvr_mape_poly smoke", {
  smoke(psvr_mape_poly(cost = 10, svm_margin = 1, degree = 2,
                       scale_factor = 1) |>
          set_engine("psvr"))
})

test_that("psvr_mape_linear smoke", {
  smoke(psvr_mape_linear(cost = 10, svm_margin = 1) |>
          set_engine("psvr"))
})

# ---- Model 2: symmetric epsilon-SVR with MAPE ----

test_that("psvr_mape_sym_rbf smoke", {
  smoke(psvr_mape_sym_rbf(cost = 10, svm_margin = 1, rbf_sigma = 1, sym_type = "even") |>
          set_engine("psvr"))
})

test_that("psvr_mape_sym_poly smoke", {
  smoke(psvr_mape_sym_poly(cost = 10, svm_margin = 1, degree = 2,
                           scale_factor = 1) |>
          set_engine("psvr", a = 1L))
})

test_that("psvr_mape_sym_linear smoke", {
  smoke(psvr_mape_sym_linear(cost = 10, svm_margin = 1) |>
          set_engine("psvr", a = 1L))
})

# ---- Model 3: LS-SVR with RMSPE ----

test_that("psvr_rmspe_rbf smoke", {
  smoke(psvr_rmspe_rbf(cost = 1000, rbf_sigma = 1) |>
          set_engine("psvr"))
})

test_that("psvr_rmspe_poly smoke", {
  smoke(psvr_rmspe_poly(cost = 1000, degree = 2, scale_factor = 1) |>
          set_engine("psvr"))
})

test_that("psvr_rmspe_linear smoke", {
  smoke(psvr_rmspe_linear(cost = 1000) |>
          set_engine("psvr"))
})

# ---- Model 4: symmetric LS-SVR with RMSPE ----

test_that("psvr_rmspe_sym_rbf smoke", {
  smoke(psvr_rmspe_sym_rbf(cost = 1000, rbf_sigma = 1, sym_type = "even") |>
          set_engine("psvr"))
})

test_that("psvr_rmspe_sym_poly smoke", {
  smoke(psvr_rmspe_sym_poly(cost = 1000, degree = 2, scale_factor = 1) |>
          set_engine("psvr", a = 1L))
})

test_that("psvr_rmspe_sym_linear smoke", {
  smoke(psvr_rmspe_sym_linear(cost = 1000) |>
          set_engine("psvr", a = 1L))
})

# ---- precondition engine-arg forwarding (RMSPE specs) -------------------
# Use a separate fixture with rho > 10 so the "auto" default activates.

set.seed(7)
n_p  <- 30
X_p  <- matrix(runif(n_p * 2, 0, 1), ncol = 2,
               dimnames = list(NULL, c("V1", "V2")))
y_p  <- pmax(1 + 99 * X_p[, 1]^2 + rnorm(n_p, sd = 0.05), 1e-3)
stopifnot(max(y_p) / min(y_p) > 10)

fit_p <- function(spec) parsnip::fit_xy(spec, x = X_p, y = y_p)

# rbf

test_that("psvr_rmspe_rbf forwards precondition='always'", {
  fit_obj <- fit_p(psvr_rmspe_rbf(cost = 1, rbf_sigma = 1) |>
                     set_engine("psvr", precondition = "always"))
  expect_true(fit_obj$fit$precondition_applied)
})

test_that("psvr_rmspe_rbf forwards precondition='never'", {
  fit_obj <- fit_p(psvr_rmspe_rbf(cost = 1, rbf_sigma = 1) |>
                     set_engine("psvr", precondition = "never"))
  expect_false(fit_obj$fit$precondition_applied)
})

# poly

test_that("psvr_rmspe_poly forwards precondition='always'", {
  fit_obj <- fit_p(psvr_rmspe_poly(cost = 1, degree = 2, scale_factor = 1) |>
                     set_engine("psvr", precondition = "always"))
  expect_true(fit_obj$fit$precondition_applied)
})

test_that("psvr_rmspe_poly forwards precondition='never'", {
  fit_obj <- fit_p(psvr_rmspe_poly(cost = 1, degree = 2, scale_factor = 1) |>
                     set_engine("psvr", precondition = "never"))
  expect_false(fit_obj$fit$precondition_applied)
})

# linear

test_that("psvr_rmspe_linear forwards precondition='always'", {
  fit_obj <- fit_p(psvr_rmspe_linear(cost = 1) |>
                     set_engine("psvr", precondition = "always"))
  expect_true(fit_obj$fit$precondition_applied)
})

test_that("psvr_rmspe_linear forwards precondition='never'", {
  fit_obj <- fit_p(psvr_rmspe_linear(cost = 1) |>
                     set_engine("psvr", precondition = "never"))
  expect_false(fit_obj$fit$precondition_applied)
})

# sym_rbf

test_that("psvr_rmspe_sym_rbf forwards precondition='always'", {
  fit_obj <- fit_p(psvr_rmspe_sym_rbf(cost = 1, rbf_sigma = 1,
                                      sym_type = "even") |>
                     set_engine("psvr", precondition = "always"))
  expect_true(fit_obj$fit$precondition_applied)
})

test_that("psvr_rmspe_sym_rbf forwards precondition='never'", {
  fit_obj <- fit_p(psvr_rmspe_sym_rbf(cost = 1, rbf_sigma = 1,
                                      sym_type = "even") |>
                     set_engine("psvr", precondition = "never"))
  expect_false(fit_obj$fit$precondition_applied)
})

# auto activates above the rho threshold (in passing)

test_that("psvr_rmspe_rbf default precondition='auto' activates at rho > 10", {
  fit_obj <- fit_p(psvr_rmspe_rbf(cost = 1, rbf_sigma = 1) |>
                     set_engine("psvr"))
  expect_true(fit_obj$fit$precondition_applied)
})
