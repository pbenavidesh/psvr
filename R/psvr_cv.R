#' Cross-validate psvr() with automatic warm-start across folds
#'
#' Fits a `psvr(loss = "mape")` model on each split in `splits`, carrying
#' the converged `(alpha, alpha_star)` from one fold into the next as the
#' SMO warm-start (Theorem 5 of arXiv:2605.01446 v3, Algorithm 1). Folds
#' are projected to feasibility before each solve. Returns a tibble with
#' one row per fold.
#'
#' This helper currently only supports `loss = "mape"`. For
#' `loss = "rmspe"` (LS-SVR), each fold is a single linear-system solve
#' with no carryover state; use `tune::tune_grid()` with parallel
#' cold-start instead.
#'
#' @param splits Either an `rsample::rset` object (e.g. from
#'   `rsample::vfold_cv()`), or a list of named lists each containing
#'   `analysis` (data frame), `assessment` (data frame), and optionally
#'   `row_ids` (integer vector of original training-row indices used for
#'   warm-start alignment across folds; defaults to positional).
#' @param ... Arguments forwarded to [psvr()]. Must specify `kernel` and
#'   the MAPE hyperparameters (`C`, `eps`). `alpha_init` and
#'   `alpha_star_init` are managed internally; supplying them via `...`
#'   is an error.
#' @param X_var Character vector of predictor column names.
#' @param y_var Single character giving the target column name.
#' @param warm_start Logical; if `FALSE`, each fold fits cold-start
#'   (useful for benchmarking the T5 speedup).
#' @param verbose Logical; if `TRUE`, print per-fold progress.
#'
#' @return A `tibble` with one row per split and columns:
#'   \describe{
#'     \item{`split_id`}{1-based fold index.}
#'     \item{`fit`}{A list-column of `psvr_fit` objects.}
#'     \item{`predictions`}{A list-column of numeric vectors (predictions
#'       on the assessment set).}
#'     \item{`metrics`}{A list-column of named numeric vectors (`mape`,
#'       `rmspe`, `mse`, `r2`).}
#'     \item{`iter_count`}{Integer; SMO iterations from
#'       `fit$solver_meta$iters`.}
#'     \item{`elapsed_sec`}{Numeric; wall-clock seconds for the fit.}
#'     \item{`warm_started`}{Logical; `TRUE` for fold > 1 when
#'       `warm_start = TRUE`.}
#'   }
#'
#' @examples
#' if (requireNamespace("rsample", quietly = TRUE) &&
#'     requireNamespace("tibble",  quietly = TRUE)) {
#'   set.seed(2026)
#'   d <- data.frame(
#'     y  = stats::rlnorm(80, sdlog = 1.0),
#'     x1 = stats::rnorm(80),
#'     x2 = stats::rnorm(80)
#'   )
#'   folds <- rsample::vfold_cv(d, v = 5)
#'   res <- psvr_cv(folds, X_var = c("x1", "x2"), y_var = "y",
#'                  loss = "mape",
#'                  kernel = make_kernel("rbf", sigma = 1),
#'                  C = 10, eps = 5)
#'   median(vapply(res$metrics, function(m) m[["mape"]], numeric(1)))
#' }
#'
#' @importFrom stats predict
#' @export
psvr_cv <- function(splits, ...,
                    X_var = NULL, y_var = NULL,
                    warm_start = TRUE, verbose = FALSE) {

  if (!requireNamespace("tibble", quietly = TRUE))
    stop("psvr_cv() requires the `tibble` package. Install it with: install.packages(\"tibble\").")

  if (is.null(X_var) || is.null(y_var))
    stop("psvr_cv() requires `X_var` (character vector of predictor column ",
         "names) and `y_var` (single target column name) so it can extract ",
         "features and target from each fold's analysis() data.")

  args <- list(...)

  if (!is.null(args$loss) && args$loss == "rmspe") {
    stop("psvr_cv() only supports `loss = \"mape\"`. For LS-SVR ",
         "(`loss = \"rmspe\"`), folds are independent (linear-system solve, ",
         "no SMO state to carry over); use `tune::tune_grid()` with ",
         "standard parallel cold-start.")
  }
  if (!is.null(args$alpha_init) || !is.null(args$alpha_star_init))
    stop("psvr_cv() manages `alpha_init` / `alpha_star_init` internally. ",
         "Do not pass them via `...`.")

  is_rset <- inherits(splits, "rset")
  if (is_rset) {
    if (!requireNamespace("rsample", quietly = TRUE))
      stop("Input is an `rsample::rset` but the rsample package is not installed.")
    n_splits <- nrow(splits)
  } else if (is.list(splits)) {
    n_splits <- length(splits)
  } else {
    stop("`splits` must be an `rsample::rset` or a list of split tuples.")
  }
  if (n_splits < 1L) stop("`splits` is empty.")

  # F6 cross-fold kernel reuse: when input is an rset (shared underlying
  # data), build Omega (or Omega_s) once over the full dataset and slice
  # per fold. The list-of-tuples path has no shared row-numbering universe,
  # so it falls back to per-fold kernel construction (per-call Rcpp
  # acceleration still applies).
  kernel_arg <- args$kernel
  sym_arg    <- args$sym
  a_val      <- if (is.null(sym_arg)) NULL else as.integer(sym_arg)
  precompute_ok <- is_rset && !is.null(kernel_arg)

  Omega_full   <- NULL
  Omega_s_full <- NULL
  if (precompute_ok) {
    data_full <- splits$splits[[1L]]$data
    X_full    <- as.matrix(data_full[, X_var, drop = FALSE])
    if (is.null(sym_arg)) {
      Omega_full   <- kernel_matrix(kernel_arg, X_full)
    } else {
      Omega_s_full <- sym_kernel_matrix(kernel_arg, X_full, a_val)
    }
  }

  results       <- vector("list", n_splits)
  fit_prev      <- NULL
  row_ids_prev  <- NULL

  for (i in seq_len(n_splits)) {
    if (is_rset) {
      split_i  <- splits$splits[[i]]
      train_i  <- rsample::analysis(split_i)
      test_i   <- rsample::assessment(split_i)
      row_ids_i <- split_i$in_id
    } else {
      train_i   <- splits[[i]]$analysis
      test_i    <- splits[[i]]$assessment
      row_ids_i <- splits[[i]]$row_ids
      if (is.null(row_ids_i)) row_ids_i <- seq_len(nrow(train_i))
    }

    X_i <- as.matrix(train_i[, X_var, drop = FALSE])
    y_i <- train_i[[y_var]]

    if (warm_start && i > 1L) {
      common <- intersect(row_ids_prev, row_ids_i)
      alpha_init      <- numeric(nrow(X_i))
      alpha_star_init <- numeric(nrow(X_i))
      # new_mask: TRUE for samples NOT in the previous fold's training set,
      # passed to Algorithm 1 Step 2 so the equality-projection shift is
      # distributed over new samples only (preserving retained values).
      new_mask <- !(row_ids_i %in% row_ids_prev)
      if (length(common) > 0L) {
        pos_in_new <- match(common, row_ids_i)
        pos_in_old <- match(common, row_ids_prev)
        alpha_init[pos_in_new]      <- fit_prev$alpha[pos_in_old]
        alpha_star_init[pos_in_new] <- fit_prev$alpha_star[pos_in_old]
      }
      warm_started <- TRUE
    } else {
      alpha_init      <- NULL
      alpha_star_init <- NULL
      new_mask        <- NULL
      warm_started    <- FALSE
    }

    precomp_args <- if (precompute_ok) {
      if (is.null(sym_arg)) {
        list(precomputed_Omega   = Omega_full[row_ids_i, row_ids_i, drop = FALSE])
      } else {
        list(precomputed_Omega_s = Omega_s_full[row_ids_i, row_ids_i, drop = FALSE])
      }
    } else {
      list()
    }

    t0 <- Sys.time()
    fit_i <- do.call(psvr, c(
      list(X = X_i, y = y_i,
           alpha_init = alpha_init,
           alpha_star_init = alpha_star_init,
           new_mask = new_mask),
      precomp_args,
      args
    ))
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    X_test <- as.matrix(test_i[, X_var, drop = FALSE])
    y_test <- test_i[[y_var]]
    preds  <- predict(fit_i, X_test)

    ss_res <- sum((y_test - preds)^2)
    ss_tot <- sum((y_test - mean(y_test))^2)
    metrics_i <- c(
      mape  = mean(abs((y_test - preds) / y_test)) * 100,
      rmspe = sqrt(mean(((y_test - preds) / y_test)^2)) * 100,
      mse   = mean((y_test - preds)^2),
      r2    = if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
    )

    iter_count <- if (is.null(fit_i$solver_meta$iters)) NA_integer_
                  else as.integer(fit_i$solver_meta$iters)

    results[[i]] <- list(
      split_id     = i,
      fit          = fit_i,
      predictions  = preds,
      metrics      = metrics_i,
      iter_count   = iter_count,
      elapsed_sec  = elapsed,
      warm_started = warm_started
    )

    if (isTRUE(verbose)) {
      cat(sprintf("Fold %d/%d: iters=%s elapsed=%.2fs warm=%s mape=%.2f\n",
                  i, n_splits,
                  if (is.na(iter_count)) "NA" else as.character(iter_count),
                  elapsed, warm_started, metrics_i[["mape"]]))
    }

    fit_prev     <- fit_i
    row_ids_prev <- row_ids_i
  }

  tibble::tibble(
    split_id     = vapply(results, `[[`, integer(1L),  "split_id"),
    fit          = lapply(results, `[[`, "fit"),
    predictions  = lapply(results, `[[`, "predictions"),
    metrics      = lapply(results, `[[`, "metrics"),
    iter_count   = vapply(results, `[[`, integer(1L),  "iter_count"),
    elapsed_sec  = vapply(results, `[[`, numeric(1L),  "elapsed_sec"),
    warm_started = vapply(results, `[[`, logical(1L),  "warm_started")
  )
}
