#' Cross-validate psvr_mape() with automatic warm-start across folds
#'
#' Fits a [psvr_mape()] model on each split in `splits`, carrying the converged
#' dual variables `(alpha, alpha_star)` from one fold into the next as the SMO
#' warm-start, so each solve starts from the previous fold's optimum instead of
#' from zero. Because consecutive folds share most of their training rows, that
#' starting point is already close to feasible; before each solve the carried
#' vectors are projected back onto the constraint set (the equality
#' \eqn{\sum_k \beta_k = 0}{sum_k beta_k = 0} and the per-sample box), with the
#' residual violation absorbed by the rows that are new to this fold. The
#' warm-start procedure is Algorithm 1 of arXiv:2605.01446 v3. Returns a tibble
#' with one row per fold.
#'
#' This helper is **MAPE-only**, and there is no `loss` argument. That is a
#' limitation of the implementation, not of the method: only [psvr_mape()] was
#' ever wired to it. LS-SVR cross-validates perfectly well, it simply has no
#' carryover state to exploit (each fold is a single linear-system solve), so
#' for [psvr_rmspe()] use `tune::tune_grid()` with parallel cold-start.
#'
#' @param splits Either an `rsample::rset` object (e.g. from
#'   `rsample::vfold_cv()`), or a list of named lists each containing
#'   `analysis` (data frame), `assessment` (data frame), and optionally
#'   `row_ids` (integer vector of original training-row indices used for
#'   warm-start alignment across folds; defaults to positional).
#' @param ... Arguments forwarded to [psvr_mape()]. Must specify `kernel` and
#'   the MAPE hyperparameters (`C`, `eps`). `alpha_init` and
#'   `alpha_star_init` are managed internally; supplying them via `...`
#'   is an error, and so is `loss`, which is not an argument of
#'   [psvr_mape()].
#' @param X_var Character vector of predictor column names.
#' @param y_var Single character giving the target column name.
#' @param warm_start Logical; if `FALSE`, each fold fits cold-start
#'   (useful for benchmarking the T5 speedup).
#' @param verbose Logical; if `TRUE`, report per-fold progress via
#'   [message()] (suppressible with [suppressMessages()]).
#'
#' @return A `tibble` with one row per split and columns:
#'   \describe{
#'     \item{`split_id`}{1-based fold index.}
#'     \item{`fit`}{A list-column of `psvr_mape` objects, or `psvr_mape_sym`
#'       when `sym_type` is `"even"` or `"odd"`.}
#'     \item{`predictions`}{A list-column of numeric vectors (predictions
#'       on the assessment set).}
#'     \item{`metrics`}{A list-column of named numeric vectors (`mape`,
#'       `rmspe`, `mse`, `r2`).}
#'     \item{`iter_count`}{Integer; SMO iterations from `fit$iterations`.}
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

  # `loss` was an argument of the superseded psvr(); psvr_mape() has no such
  # formal, so forwarding it would fail with an opaque dots error. Reject it
  # here with the reason. The wording is deliberately NOT YET rather than
  # NEVER: psvr_cv() is MAPE-only because only .fit_mape() was ever wired to
  # it, not because RMSPE resists cross-validation -- LS-SVR is a linear
  # system and cross-validates perfectly well. Extending it will add a
  # function or an argument, not restore this one.
  if ("loss" %in% names(args)) {
    stop("`psvr_cv()` currently supports MAPE only; `loss` is not an ",
         "argument. RMSPE cross-validation is not implemented -- see ",
         "PSVR_STATUS.md. For LS-SVR today, use `tune::tune_grid()` with ",
         "standard parallel cold-start; folds are independent there (a ",
         "linear-system solve, no SMO state to carry over).",
         call. = FALSE)
  }
  if (!is.null(args$alpha_init) || !is.null(args$alpha_star_init))
    stop("psvr_cv() manages `alpha_init` / `alpha_star_init` internally. ",
         "Do not pass them via `...`.")

  # F7 — informational note about T5 (warm-start) × T7 (block-k=4) interaction.
  # The two theorems do not compose multiplicatively under CV: T7 dominates
  # the per-fold iter reduction (~50% on the snapshot fixture), leaving only
  # a small warm-start perturbation cost (~3% on N=300, rho_y=2388 fixture).
  # For pure-F5 warm-start behavior, set block_k4_enabled = FALSE.
  # See paper TODO #10 for the empirical calibration.
  block_k4_active <- if (is.null(args$block_k4_enabled)) TRUE
                     else isTRUE(args$block_k4_enabled)
  if (isTRUE(verbose) && isTRUE(warm_start) && block_k4_active) {
    message("Note: Theorem 5 (warm-start) and Theorem 7 (block-k=4) interact; ",
            "warm-start advantage is dominated by block-k=4 gains. For pure ",
            "F5 warm-start behavior, set block_k4_enabled = FALSE.")
  }

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
  # `sym_type` is the public vocabulary on psvr_mape(); `a` is the internal
  # integer. Absent means "none", the psvr_mape() default.
  sym_type_arg <- if (is.null(args$sym_type)) "none" else args$sym_type
  a_val      <- if (identical(sym_type_arg, "none")) NULL
                else .sym_type_to_a(sym_type_arg)
  precompute_ok <- is_rset && !is.null(kernel_arg)

  Omega_full   <- NULL
  Omega_s_full <- NULL
  if (precompute_ok) {
    data_full <- splits$splits[[1L]]$data
    X_full    <- as.matrix(data_full[, X_var, drop = FALSE])
    if (is.null(a_val)) {
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
      warm_started    <- FALSE
    }

    precomp_args <- if (precompute_ok) {
      if (is.null(a_val)) {
        list(precomputed_Omega   = Omega_full[row_ids_i, row_ids_i, drop = FALSE])
      } else {
        list(precomputed_Omega_s = Omega_s_full[row_ids_i, row_ids_i, drop = FALSE])
      }
    } else {
      list()
    }

    t0 <- Sys.time()
    fit_i <- do.call(psvr_mape, c(
      list(X = X_i, y = y_i,
           alpha_init = alpha_init,
           alpha_star_init = alpha_star_init),
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

    iter_count <- if (is.null(fit_i$iterations)) NA_integer_
                  else as.integer(fit_i$iterations)

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
      message(sprintf("Fold %d/%d: iters=%s elapsed=%.2fs warm=%s mape=%.2f",
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
