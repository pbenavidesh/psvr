devtools::load_all("C:/Users/behep/OneDrive - ITESO/PhD/00-Tesis/psvr",
                   recompile = TRUE, quiet = TRUE)

if (!requireNamespace("rsample", quietly = TRUE)) stop("rsample required")
if (!requireNamespace("tibble",  quietly = TRUE)) stop("tibble required")

set.seed(2026)
N <- 60
d <- data.frame(
  y  = rlnorm(N, sdlog = 0.5),
  x1 = rnorm(N),
  x2 = rnorm(N),
  x3 = rnorm(N)
)
folds <- rsample::vfold_cv(d, v = 5)
K <- make_kernel("rbf", sigma = 1)

# Path 1: psvr_cv() with cross-fold precompute (F6 default)
res_cv <- psvr_cv(folds, X_var = c("x1","x2","x3"), y_var = "y",
                  loss = "mape", kernel = K, C = 10, eps = 5,
                  warm_start = TRUE, verbose = FALSE)

# Path 2: manual per-fold loop WITHOUT precomputed_Omega.
# Reproduces the warm-start handoff manually.
results_manual <- vector("list", nrow(folds))
fit_prev     <- NULL
row_ids_prev <- NULL
for (i in seq_len(nrow(folds))) {
  split_i <- folds$splits[[i]]
  train_i <- rsample::analysis(split_i)
  X_i     <- as.matrix(train_i[, c("x1","x2","x3"), drop = FALSE])
  y_i     <- train_i$y
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

  fit_i <- psvr(X_i, y_i, loss = "mape", kernel = K, C = 10, eps = 5,
                alpha_init = alpha_init,
                alpha_star_init = alpha_star_init,
                new_mask = new_mask)
  results_manual[[i]] <- fit_i
  fit_prev     <- fit_i
  row_ids_prev <- row_ids_i
}

# Compare fits per fold
for (i in seq_along(results_manual)) {
  fit_cv_i  <- res_cv$fit[[i]]
  fit_man_i <- results_manual[[i]]
  cat(sprintf("Fold %d: alpha %s, beta %s, b %s, iters %s == %s\n",
              i,
              identical(fit_cv_i$alpha, fit_man_i$alpha),
              identical(fit_cv_i$beta,  fit_man_i$beta),
              identical(fit_cv_i$b,     fit_man_i$b),
              fit_cv_i$solver_meta$iters,
              fit_man_i$solver_meta$iters))
}

# Symmetric variant
cat("\n--- Symmetric (sym = +1) ---\n")
res_cv_sym <- psvr_cv(folds, X_var = c("x1","x2","x3"), y_var = "y",
                      loss = "mape", sym = +1L, kernel = K, C = 10, eps = 5,
                      warm_start = TRUE, verbose = FALSE)

results_manual_sym <- vector("list", nrow(folds))
fit_prev <- NULL; row_ids_prev <- NULL
for (i in seq_len(nrow(folds))) {
  split_i <- folds$splits[[i]]
  train_i <- rsample::analysis(split_i)
  X_i     <- as.matrix(train_i[, c("x1","x2","x3"), drop = FALSE])
  y_i     <- train_i$y
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
  fit_i <- psvr(X_i, y_i, loss = "mape", sym = +1L, kernel = K, C = 10, eps = 5,
                alpha_init = alpha_init,
                alpha_star_init = alpha_star_init,
                new_mask = new_mask)
  results_manual_sym[[i]] <- fit_i
  fit_prev <- fit_i; row_ids_prev <- row_ids_i
}

for (i in seq_along(results_manual_sym)) {
  fit_cv_i  <- res_cv_sym$fit[[i]]
  fit_man_i <- results_manual_sym[[i]]
  cat(sprintf("Fold %d: alpha %s, beta %s, b %s, iters %s == %s\n",
              i,
              identical(fit_cv_i$alpha, fit_man_i$alpha),
              identical(fit_cv_i$beta,  fit_man_i$beta),
              identical(fit_cv_i$b,     fit_man_i$b),
              fit_cv_i$solver_meta$iters,
              fit_man_i$solver_meta$iters))
}
