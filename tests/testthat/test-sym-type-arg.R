## Stage-1 tests for the `sym_type` model argument on the six non-symmetric
## parsnip specs.
##
## THIS FILE IS DELETED IN STAGE 2 BY DESIGN. Its equivalence tests exist only
## while the old `_sym_` specs and the new `sym_type` argument coexist: they
## prove, BEFORE stage 2 repoints the twelve goldens, that the new argument
## reproduces the old specs' numerics exactly. Once the `_sym_` specs are gone
## there is nothing left to compare against, and the repointed goldens in
## test-bit-identical.R carry the guarantee instead.

library(parsnip)

# ---- Fixture -------------------------------------------------------------
# VERBATIM COPY of test-bit-identical.R:15-37. The equivalence tests below are
# meaningful only if this fixture is byte-identical to the one the protected
# goldens run on (_snaps/bit-identical.md, MD5 46a4fa24). If that range
# changes, change this block to match or delete these tests -- do not let them
# drift. Verify with:
#   diff <(sed -n '15,37p' tests/testthat/test-bit-identical.R) \
#        <(sed -n '21,43p' tests/testthat/test-sym-type-arg.R)
make_fixture <- function() {
  set.seed(2026)
  X      <- matrix(stats::rnorm(50 * 5), 50, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  y      <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
  X_test <- matrix(stats::rnorm(20 * 5), 20, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  list(X       = X,
       y       = y,
       X_test  = X_test,
       df_test = as.data.frame(X_test))
}

# Hyperparameters fixed across all tests.
HP <- list(
  C            = 10,
  eps          = 5,
  gamma        = 100,
  rbf_sigma    = 1,
  degree       = 2L,
  scale_factor = 1,
  a            = 1L
)

# Same helper as test-bit-identical.R:101-104.
fit_and_predict <- function(spec, fx) {
  fit_obj <- parsnip::fit_xy(spec, x = fx$X, y = fx$y)
  predict(fit_obj, new_data = fx$df_test)$.pred
}


# ---- (a) Six equivalence tests -------------------------------------------
# new spec + sym_type = "even"  ==  old _sym_ spec at a = 1L, exactly.
#
# The MAPE poly and linear pairs do not converge within max_iter (see the
# "Known limitation" note in CLAUDE.md: the SMO solver stalls on linear and
# polynomial kernels with MAPE loss). Both sides stall at the SAME endpoint,
# so tolerance = 0 remains the right assertion -- it proves the two code paths
# are one path, which is exactly the claim. suppressWarnings() swallows the
# expected "did not converge" notice.

test_that("mape_rbf sym_type='even' == psvr_mape_sym_rbf", {
  skip_on_cran()
  fx  <- make_fixture()
  new <- fit_and_predict(
    psvr_mape_rbf(cost = HP$C, svm_margin = HP$eps,
                  rbf_sigma = HP$rbf_sigma, sym_type = "even") |>
      set_engine("psvr"), fx)
  old <- fit_and_predict(
    psvr_mape_sym_rbf(cost = HP$C, svm_margin = HP$eps,
                      rbf_sigma = HP$rbf_sigma, sym_type = "even") |>
      set_engine("psvr"), fx)
  expect_equal(new, old, tolerance = 0)
})

test_that("mape_poly sym_type='even' == psvr_mape_sym_poly", {
  skip_on_cran()
  fx  <- make_fixture()
  new <- suppressWarnings(fit_and_predict(
    psvr_mape_poly(cost = HP$C, svm_margin = HP$eps, degree = HP$degree,
                   scale_factor = HP$scale_factor, sym_type = "even") |>
      set_engine("psvr"), fx))
  old <- suppressWarnings(fit_and_predict(
    psvr_mape_sym_poly(cost = HP$C, svm_margin = HP$eps, degree = HP$degree,
                       scale_factor = HP$scale_factor) |>
      set_engine("psvr", a = HP$a), fx))
  expect_equal(new, old, tolerance = 0)
})

test_that("mape_linear sym_type='even' == psvr_mape_sym_linear", {
  skip_on_cran()
  fx  <- make_fixture()
  new <- suppressWarnings(fit_and_predict(
    psvr_mape_linear(cost = HP$C, svm_margin = HP$eps, sym_type = "even") |>
      set_engine("psvr"), fx))
  old <- suppressWarnings(fit_and_predict(
    psvr_mape_sym_linear(cost = HP$C, svm_margin = HP$eps) |>
      set_engine("psvr", a = HP$a), fx))
  expect_equal(new, old, tolerance = 0)
})

test_that("rmspe_rbf sym_type='even' == psvr_rmspe_sym_rbf", {
  skip_on_cran()
  fx  <- make_fixture()
  new <- fit_and_predict(
    psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma,
                   sym_type = "even") |>
      set_engine("psvr"), fx)
  old <- fit_and_predict(
    psvr_rmspe_sym_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma,
                       sym_type = "even") |>
      set_engine("psvr"), fx)
  expect_equal(new, old, tolerance = 0)
})

test_that("rmspe_poly sym_type='even' == psvr_rmspe_sym_poly", {
  skip_on_cran()
  fx  <- make_fixture()
  new <- fit_and_predict(
    psvr_rmspe_poly(cost = HP$gamma, degree = HP$degree,
                    scale_factor = HP$scale_factor, sym_type = "even") |>
      set_engine("psvr"), fx)
  old <- fit_and_predict(
    psvr_rmspe_sym_poly(cost = HP$gamma, degree = HP$degree,
                        scale_factor = HP$scale_factor) |>
      set_engine("psvr", a = HP$a), fx)
  expect_equal(new, old, tolerance = 0)
})

test_that("rmspe_linear sym_type='even' == psvr_rmspe_sym_linear", {
  skip_on_cran()
  fx  <- make_fixture()
  new <- fit_and_predict(
    psvr_rmspe_linear(cost = HP$gamma, sym_type = "even") |>
      set_engine("psvr"), fx)
  old <- fit_and_predict(
    psvr_rmspe_sym_linear(cost = HP$gamma) |>
      set_engine("psvr", a = HP$a), fx)
  expect_equal(new, old, tolerance = 0)
})


# ---- (b) sym_type = tune() resolves on ALL SIX survivors ------------------
# This is the executable proof that PSVR_STATUS.md section 9.2 is closed.
# Before this change, `sym_type` was a registered model argument on exactly two
# specs (psvr_mape_sym_rbf, psvr_rmspe_sym_rbf); the four poly/linear symmetric
# specs took a raw `a` as an ENGINE argument, which tune() cannot reach. The
# poly and linear rows below are the ones that could not have passed before.

sym_specs <- list(
  psvr_mape_rbf     = function() psvr_mape_rbf(cost = 10, svm_margin = 1,
                                               rbf_sigma = 1,
                                               sym_type = tune::tune()),
  psvr_mape_poly    = function() psvr_mape_poly(cost = 10, svm_margin = 1,
                                                degree = 2, scale_factor = 1,
                                                sym_type = tune::tune()),
  psvr_mape_linear  = function() psvr_mape_linear(cost = 10, svm_margin = 1,
                                                  sym_type = tune::tune()),
  psvr_rmspe_rbf    = function() psvr_rmspe_rbf(cost = 1000, rbf_sigma = 1,
                                                sym_type = tune::tune()),
  psvr_rmspe_poly   = function() psvr_rmspe_poly(cost = 1000, degree = 2,
                                                 scale_factor = 1,
                                                 sym_type = tune::tune()),
  psvr_rmspe_linear = function() psvr_rmspe_linear(cost = 1000,
                                                   sym_type = tune::tune())
)

for (nm in names(sym_specs)) {
  local({
    nm_local   <- nm
    make_local <- sym_specs[[nm]]

    test_that(sprintf("sym_type = tune() resolves through dials on %s()",
                      nm_local), {
      skip_if_not_installed("tune")
      skip_if_not_installed("dials")

      spec <- make_local() |> parsnip::set_engine("psvr")
      pset <- tune::extract_parameter_set_dials(spec)

      expect_true("sym_type" %in% pset$id)

      p <- dials::extract_parameter_dials(pset, "sym_type")
      expect_s3_class(p, "qual_param")
      expect_setequal(p$values, c("none", "even", "odd"))
    })
  })
}


# ---- (c) .sym_type_to_a() validation -------------------------------------

test_that(".sym_type_to_a maps the two symmetric levels", {
  expect_identical(psvr:::.sym_type_to_a("even"),  1L)
  expect_identical(psvr:::.sym_type_to_a("odd"),  -1L)
})

test_that(".sym_type_to_a rejects invalid values instead of defaulting to odd", {
  # The pre-stage-1 form was `if (sym_type == "even") 1L else -1L`, which
  # silently mapped every one of these to a = -1L.
  expect_error(psvr:::.sym_type_to_a("EVEN"),   "sym_type")
  expect_error(psvr:::.sym_type_to_a("evne"),   "sym_type")
  expect_error(psvr:::.sym_type_to_a(""),       "sym_type")
  # "none" selects a different fitter and is branched before this helper, so
  # reaching it here is a caller bug and must be reported as one.
  expect_error(psvr:::.sym_type_to_a("none"),   "sym_type")
  expect_error(psvr:::.sym_type_to_a(1L),       "sym_type")
  expect_error(psvr:::.sym_type_to_a(NA),       "sym_type")
  expect_error(psvr:::.sym_type_to_a(c("even", "odd")), "sym_type")
})

test_that("sym_type_param() validates its values argument", {
  expect_setequal(sym_type_param()$values, c("none", "even", "odd"))
  expect_setequal(sym_type_param(values = c("even", "odd"))$values,
                  c("even", "odd"))
  expect_error(sym_type_param(values = c("even", "od")))
  expect_error(sym_type_param(values = "symmetric"))
})


# ---- (d) omitted == NULL == "none" == the pre-stage-1 fit path ------------
# Pins the mechanism the whole stage rests on: parsnip drops NULL-quosure args
# before building the fit call (parsnip:::translate.default filters on
# null_value(); parsnip:::make_call discards NULL), so an unset sym_type never
# reaches the wrapper and the wrapper's own "none" default applies.

test_that("mape_rbf: omitted == NULL == 'none' == direct .fit_mape", {
  skip_on_cran()
  fx <- make_fixture()

  omitted <- fit_and_predict(
    psvr_mape_rbf(cost = HP$C, svm_margin = HP$eps,
                  rbf_sigma = HP$rbf_sigma) |> set_engine("psvr"), fx)
  explicit_null <- fit_and_predict(
    psvr_mape_rbf(cost = HP$C, svm_margin = HP$eps,
                  rbf_sigma = HP$rbf_sigma, sym_type = NULL) |>
      set_engine("psvr"), fx)
  explicit_none <- fit_and_predict(
    psvr_mape_rbf(cost = HP$C, svm_margin = HP$eps,
                  rbf_sigma = HP$rbf_sigma, sym_type = "none") |>
      set_engine("psvr"), fx)

  direct <- predict(
    psvr:::.fit_mape(X = fx$X, y = fx$y,
                     kernel = make_kernel("rbf", sigma = HP$rbf_sigma),
                     C = HP$C, eps = HP$eps),
    fx$X_test)

  expect_equal(explicit_null, omitted, tolerance = 0)
  expect_equal(explicit_none, omitted, tolerance = 0)
  expect_equal(as.numeric(direct), omitted, tolerance = 0)
})

test_that("rmspe_rbf: omitted == NULL == 'none' == direct .fit_rmspe", {
  skip_on_cran()
  fx <- make_fixture()

  omitted <- fit_and_predict(
    psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma) |>
      set_engine("psvr"), fx)
  explicit_null <- fit_and_predict(
    psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma,
                   sym_type = NULL) |> set_engine("psvr"), fx)
  explicit_none <- fit_and_predict(
    psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma,
                   sym_type = "none") |> set_engine("psvr"), fx)

  direct <- predict(
    psvr:::.fit_rmspe(X = fx$X, y = fx$y,
                      kernel = make_kernel("rbf", sigma = HP$rbf_sigma),
                      gamma = HP$gamma),
    fx$X_test)

  expect_equal(explicit_null, omitted, tolerance = 0)
  expect_equal(explicit_none, omitted, tolerance = 0)
  expect_equal(as.numeric(direct), omitted, tolerance = 0)
})
