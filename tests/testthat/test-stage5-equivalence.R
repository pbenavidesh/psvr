## TEMPORARY -- API-redesign stage 5, deleted in the same commit that deletes
## psvr(). Do not repoint, do not extend, do not keep.
##
## Between stage 5a (psvr_mape() / psvr_rmspe() added) and stage 5c (psvr()
## deleted) both APIs exist. That window is the only chance to prove the split
## is a pure re-spelling DIRECTLY, at tolerance = 0. The goldens prove it only
## indirectly: if an MD5 moved afterwards they would say "something changed"
## without saying which of the eighteen calls. These say which.
##
## Stage 1 of this redesign set the pattern (six new-vs-old spec equivalence
## tests, deleted by design once the old specs went, output recorded in the
## commit body). Retrieval after deletion:
##   git show <sha>:tests/testthat/test-stage5-equivalence.R
##
## Coverage: {mape, rmspe} x {none, even, odd} x {rbf, poly, linear} = 18
## pairs in 6 test_that() blocks. The "odd" level is coverage the goldens do
## NOT have -- test-bit-identical.R and test-psvr-direct.R both pin a = 1L
## only, so sym = -1L / sym_type = "odd" has never been compared across the
## two entry points by anything else.

## ---- Fixture: copied VERBATIM from test-bit-identical.R -------------------
## It must stay byte-identical to that copy -- same seed, same dimnames, same
## hyperparameters -- so that any difference seen here is attributable to the
## entry point and to nothing else. Copied rather than sourced because every
## test file in this suite is self-contained.
##
## VERIFIED 2026-08-10, MD5 9303f430962d7f92aa1b57d1b16b922c on both sides.
## To re-verify (the two comment lines between the blocks differ; the code
## does not):
##   { sed -n '15,26p;29,37p' tests/testthat/test-bit-identical.R; } | md5sum
##   { sed -n '26,37p;39,47p' tests/testthat/test-stage5-equivalence.R; } | md5sum
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

HP <- list(
  C            = 10,
  eps          = 5,
  gamma        = 100,
  rbf_sigma    = 1,
  degree       = 2L,
  scale_factor = 1,
  a            = 1L
)

## ---- Kernels: the three the goldens cover --------------------------------
KERNELS <- list(
  rbf    = function() make_kernel("rbf", sigma = HP$rbf_sigma),
  poly   = function() make_kernel("polynomial", degree = HP$degree,
                                  coef0 = HP$scale_factor),
  linear = function() make_kernel("linear")
)

## sym_type ("none"/"even"/"odd") is the new vocabulary; sym (NULL/+1L/-1L) is
## the old one. This mapping IS the claim under test on the symmetry axis.
SYM <- list(none = NULL, even = 1L, odd = -1L)

## Compare one (loss, sym_type) cell across all three kernels.
##
## suppressWarnings() is required on the poly and linear MAPE cells: the
## documented SMO non-convergence pathology fires there, on BOTH sides.
## tolerance = 0 remains exactly right in that case -- two solvers that stall
## at bit-identical non-converged endpoints have taken bit-identical code
## paths, which is the whole claim. A stall that differed would fail here.
expect_entry_points_agree <- function(loss, sym_type) {
  fx  <- make_fixture()
  sym <- SYM[[sym_type]]

  for (kname in names(KERNELS)) {
    K <- KERNELS[[kname]]()

    if (loss == "mape") {
      old <- suppressWarnings(
        psvr(fx$X, fx$y, loss = "mape", sym = sym, kernel = K,
             C = HP$C, eps = HP$eps))
      new <- suppressWarnings(
        psvr_mape(fx$X, fx$y, sym_type = sym_type, kernel = K,
                  C = HP$C, eps = HP$eps))
    } else {
      old <- suppressWarnings(
        psvr(fx$X, fx$y, loss = "rmspe", sym = sym, kernel = K,
             gamma = HP$gamma))
      new <- suppressWarnings(
        psvr_rmspe(fx$X, fx$y, sym_type = sym_type, kernel = K,
                   gamma = HP$gamma))
    }

    expect_equal(
      predict(new, fx$X_test),
      predict(old, fx$X_test),
      tolerance = 0,
      label = sprintf("psvr_%s(sym_type = '%s') / %s predictions",
                      loss, sym_type, kname)
    )
  }
}

## ---- 6 blocks x 3 kernels = 18 assertions --------------------------------

test_that("psvr_mape(sym_type = 'none') matches psvr(loss = 'mape')", {
  skip_on_cran()
  expect_entry_points_agree("mape", "none")
})

test_that("psvr_mape(sym_type = 'even') matches psvr(loss = 'mape', sym = +1L)", {
  skip_on_cran()
  expect_entry_points_agree("mape", "even")
})

test_that("psvr_mape(sym_type = 'odd') matches psvr(loss = 'mape', sym = -1L)", {
  skip_on_cran()
  expect_entry_points_agree("mape", "odd")
})

test_that("psvr_rmspe(sym_type = 'none') matches psvr(loss = 'rmspe')", {
  skip_on_cran()
  expect_entry_points_agree("rmspe", "none")
})

test_that("psvr_rmspe(sym_type = 'even') matches psvr(loss = 'rmspe', sym = +1L)", {
  skip_on_cran()
  expect_entry_points_agree("rmspe", "even")
})

test_that("psvr_rmspe(sym_type = 'odd') matches psvr(loss = 'rmspe', sym = -1L)", {
  skip_on_cran()
  expect_entry_points_agree("rmspe", "odd")
})
