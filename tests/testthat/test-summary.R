# summary() on the four fit classes.
#
# Before API-redesign stage 5 the generic was registered on psvr_fit only
# (NAMESPACE:28, the single summary entry), so it was unreachable from
# parsnip::extract_fit_engine() and had NO test coverage at all -- the same
# property coef.psvr_fit had before 0.0.2.9011, and the reason that defect
# went unnoticed. Both are closed now.
#
# expect_output(), deliberately NOT expect_snapshot(): a fifth _snaps/*.md file
# sitting beside the four protected F7.5 baselines invites exactly the
# confusion the MD5 gate exists to prevent. Nothing here needs byte-exact
# output; it needs the right facts on the right lines.

set.seed(11)
X_tr <- matrix(runif(60, 0.5, 3), 20, 3)
y_tr <- 2 + X_tr[, 1]^2 + abs(rnorm(20, 0, 0.1))
K    <- make_kernel("rbf", sigma = 1)

test_that("summary.psvr_mape reports kernel, counts, hyperparameters and SMO state", {
  fit <- psvr_mape(X_tr, y_tr, kernel = K, C = 10, eps = 5)
  expect_output(summary(fit), "Epsilon-SVR with MAPE loss\\s+\\[psvr_mape\\]")
  expect_output(summary(fit), "Kernel:\\s+RBF \\(sigma = 1\\)")
  expect_output(summary(fit), "Training obs\\.:\\s+20")
  expect_output(summary(fit), "Support vectors:\\s+\\d+ \\(\\d+\\.\\d%\\)")
  expect_output(summary(fit), "C\\s+= 10")
  expect_output(summary(fit), "eps\\s+= 5")
  # iterations + convergence status replace the psvr_fit "Solver:" line, which
  # no fit class can reproduce: none of them records which backend ran.
  expect_output(summary(fit), "SMO iterations:\\s+\\d+ \\(converged\\)")
  # No symmetry line on the non-symmetric class. Asserted over the captured
  # lines rather than with a negative-lookahead regex: expect_output() matches
  # against one multi-line string, where an anchored (?!...) does not mean
  # what it looks like it means.
  expect_false(any(grepl("Symmetry:", capture.output(summary(fit)))))
})

test_that("summary.psvr_mape_sym adds the symmetry line", {
  even <- psvr_mape(X_tr, y_tr, sym_type = "even", kernel = K, C = 10, eps = 5)
  odd  <- psvr_mape(X_tr, y_tr, sym_type = "odd",  kernel = K, C = 10, eps = 5)
  expect_output(summary(even), "\\[psvr_mape_sym\\]")
  expect_output(summary(even), "Symmetry:\\s+even\\s+\\(a = 1\\)")
  expect_output(summary(odd),  "Symmetry:\\s+odd\\s+\\(a = -1\\)")
  expect_output(summary(even), "C\\s+= 10")
})

test_that("summary.psvr_rmspe reports Gamma and the preconditioner, and no SV count", {
  fit <- psvr_rmspe(X_tr, y_tr, kernel = K, gamma = 100)
  expect_output(summary(fit), "LS-SVR with RMSPE loss\\s+\\[psvr_rmspe\\]")
  expect_output(summary(fit), "Training obs\\.:\\s+20")
  expect_output(summary(fit), "Gamma\\s+= 100")
  expect_output(summary(fit), "Preconditioner:\\s+(applied|not applied)")
  # LS-SVR does no pruning, so a support-vector count would be meaningless:
  # every training point contributes.
  out <- capture.output(summary(fit))
  expect_false(any(grepl("Support vectors:", out)))
  expect_false(any(grepl("Symmetry:", out)))
})

test_that("summary.psvr_rmspe_sym adds the symmetry line", {
  even <- psvr_rmspe(X_tr, y_tr, sym_type = "even", kernel = K, gamma = 100)
  odd  <- psvr_rmspe(X_tr, y_tr, sym_type = "odd",  kernel = K, gamma = 100)
  expect_output(summary(even), "\\[psvr_rmspe_sym\\]")
  expect_output(summary(even), "Symmetry:\\s+even\\s+\\(a = 1\\)")
  expect_output(summary(odd),  "Symmetry:\\s+odd\\s+\\(a = -1\\)")
  expect_output(summary(even), "Gamma\\s+= 100")
})

test_that("summary() returns its input invisibly, on every class", {
  fits <- list(
    psvr_mape(X_tr, y_tr, kernel = K, C = 10, eps = 5),
    psvr_mape(X_tr, y_tr, sym_type = "even", kernel = K, C = 10, eps = 5),
    psvr_rmspe(X_tr, y_tr, kernel = K, gamma = 100),
    psvr_rmspe(X_tr, y_tr, sym_type = "even", kernel = K, gamma = 100)
  )
  for (f in fits) {
    out <- capture.output(res <- summary(f))
    expect_identical(res, f)
    expect_true(length(out) > 0L)
  }
})

test_that("summary() is reachable through parsnip, which it was not before", {
  # The whole point of registering it on the four fit classes: a parsnip user
  # unwrapping with extract_fit_engine() now gets a summary(). Under psvr_fit
  # the generic existed but no parsnip fit could ever dispatch to it.
  df   <- data.frame(x1 = X_tr[, 1], x2 = X_tr[, 2], x3 = X_tr[, 3], y = y_tr)
  spec <- psvr_rmspe_rbf(cost = 100, rbf_sigma = 1)
  pfit <- parsnip::fit(spec, y ~ x1 + x2 + x3, data = df)
  eng  <- parsnip::extract_fit_engine(pfit)
  expect_s3_class(eng, "psvr_rmspe")
  expect_output(summary(eng), "LS-SVR with RMSPE loss")
})
