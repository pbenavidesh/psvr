skip_if_not_installed("rsample")
skip_if_not_installed("tibble")

# Helper: manual per-fold loop reproducing psvr_cv()'s warm-start handoff
# WITHOUT the precomputed_Omega path. The two paths should produce
# bit-identical fits when the C++ kernel dispatch is bit-identical to the
# legacy R kernel.
manual_cv <- function(folds, X_var, y_var, ...) {
  args <- list(...)
  results      <- vector("list", nrow(folds))
  fit_prev     <- NULL
  row_ids_prev <- NULL
  for (i in seq_len(nrow(folds))) {
    split_i  <- folds$splits[[i]]
    train_i  <- rsample::analysis(split_i)
    X_i      <- as.matrix(train_i[, X_var, drop = FALSE])
    y_i      <- train_i[[y_var]]
    row_ids_i <- split_i$in_id

    if (i > 1L) {
      common <- intersect(row_ids_prev, row_ids_i)
      alpha_init      <- numeric(nrow(X_i))
      alpha_star_init <- numeric(nrow(X_i))
      new_mask        <- !(row_ids_i %in% row_ids_prev)
      if (length(common) > 0L) {
        pos_in_new <- match(common, row_ids_i)
        pos_in_old <- match(common, row_ids_prev)
        alpha_init[pos_in_new]      <- fit_prev$alpha[pos_in_old]
        alpha_star_init[pos_in_new] <- fit_prev$alpha_star[pos_in_old]
      }
    } else {
      alpha_init <- NULL; alpha_star_init <- NULL; new_mask <- NULL
    }
    fit_i <- do.call(psvr, c(
      list(X = X_i, y = y_i,
           alpha_init = alpha_init,
           alpha_star_init = alpha_star_init,
           new_mask = new_mask),
      args
    ))
    results[[i]]   <- fit_i
    fit_prev       <- fit_i
    row_ids_prev   <- row_ids_i
  }
  results
}

test_that("psvr_cv() precomputed path produces bit-identical fits (non-sym)", {
  set.seed(2026)
  N <- 50
  d <- data.frame(
    y  = rlnorm(N, sdlog = 0.5),
    x1 = rnorm(N),
    x2 = rnorm(N),
    x3 = rnorm(N)
  )
  folds <- rsample::vfold_cv(d, v = 5)
  K <- make_kernel("rbf", sigma = 1)

  res_cv  <- suppressWarnings(psvr_cv(folds,
                                      X_var = c("x1", "x2", "x3"), y_var = "y",
                                      loss = "mape", kernel = K, C = 10, eps = 5,
                                      warm_start = TRUE, verbose = FALSE))
  res_man <- suppressWarnings(manual_cv(folds,
                                        X_var = c("x1", "x2", "x3"), y_var = "y",
                                        loss = "mape", kernel = K, C = 10, eps = 5))

  for (i in seq_along(res_man)) {
    expect_identical(res_cv$fit[[i]]$alpha,      res_man[[i]]$alpha)
    expect_identical(res_cv$fit[[i]]$alpha_star, res_man[[i]]$alpha_star)
    expect_identical(res_cv$fit[[i]]$beta,       res_man[[i]]$beta)
    expect_identical(res_cv$fit[[i]]$b,          res_man[[i]]$b)
    expect_identical(res_cv$fit[[i]]$solver_meta$iters,
                     res_man[[i]]$solver_meta$iters)
    expect_identical(res_cv$fit[[i]]$solver_meta$converged,
                     res_man[[i]]$solver_meta$converged)
  }
})

test_that("psvr_cv() precomputed path produces bit-identical fits (sym = +1)", {
  set.seed(2026)
  N <- 50
  d <- data.frame(
    y  = rlnorm(N, sdlog = 0.5),
    x1 = rnorm(N),
    x2 = rnorm(N)
  )
  folds <- rsample::vfold_cv(d, v = 4)
  K <- make_kernel("rbf", sigma = 1)

  res_cv  <- suppressWarnings(psvr_cv(folds,
                                      X_var = c("x1", "x2"), y_var = "y",
                                      loss = "mape", sym = +1L,
                                      kernel = K, C = 10, eps = 5,
                                      warm_start = TRUE, verbose = FALSE))
  res_man <- suppressWarnings(manual_cv(folds,
                                        X_var = c("x1", "x2"), y_var = "y",
                                        loss = "mape", sym = +1L,
                                        kernel = K, C = 10, eps = 5))

  for (i in seq_along(res_man)) {
    expect_identical(res_cv$fit[[i]]$alpha,      res_man[[i]]$alpha)
    expect_identical(res_cv$fit[[i]]$beta,       res_man[[i]]$beta)
    expect_identical(res_cv$fit[[i]]$b,          res_man[[i]]$b)
    expect_identical(res_cv$fit[[i]]$solver_meta$iters,
                     res_man[[i]]$solver_meta$iters)
  }
})

test_that("psvr_cv() list-of-tuples path still works (no precompute)", {
  set.seed(2026)
  N <- 40
  d <- data.frame(
    y  = rlnorm(N, sdlog = 0.5),
    x1 = rnorm(N),
    x2 = rnorm(N)
  )
  folds_rset <- rsample::vfold_cv(d, v = 4)
  # Manually convert to list-of-tuples (forces the no-precompute branch)
  folds_list <- lapply(folds_rset$splits, function(s) {
    list(analysis   = rsample::analysis(s),
         assessment = rsample::assessment(s),
         row_ids    = s$in_id)
  })
  K <- make_kernel("rbf", sigma = 1)

  res_list <- suppressWarnings(psvr_cv(folds_list,
                                       X_var = c("x1", "x2"), y_var = "y",
                                       loss = "mape", kernel = K, C = 10, eps = 5,
                                       warm_start = TRUE))
  res_rset <- suppressWarnings(psvr_cv(folds_rset,
                                       X_var = c("x1", "x2"), y_var = "y",
                                       loss = "mape", kernel = K, C = 10, eps = 5,
                                       warm_start = TRUE))

  # Both paths should agree bit-exactly because kernel_matrix() is
  # bit-identical between the slice-from-full path and the per-fold
  # rebuild path.
  for (i in seq_len(nrow(folds_rset))) {
    expect_identical(res_list$fit[[i]]$alpha, res_rset$fit[[i]]$alpha)
    expect_identical(res_list$fit[[i]]$b,     res_rset$fit[[i]]$b)
  }
})
