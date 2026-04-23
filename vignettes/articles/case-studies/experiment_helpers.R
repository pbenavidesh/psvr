# _experiment_helpers.R
# 2026-04-23
# Single source of truth for model definitions, metric functions, and
# experiment protocol across all psvr cross-section case-study articles.
# Do NOT call library() here — callers are responsible for loading packages.

# ── SECTION 1: Model label lookup table ──────────────────────────────────────

MODEL_LABELS <- tibble::tibble(
  id = c(
    "m1", "m2",
    "m3a", "m3b",
    "m4a", "m4b",
    "b1", "b2", "b3", "b4", "b5", "b6", "b7a"
  ),
  label = c(
    "SVR-MAPE",
    "SVR-MAPE + Sym. Kernel",
    "LS-RMSPE (MAPE opt.)",
    "LS-RMSPE (RMSPE opt.)",
    "LS-RMSPE + Sym. Kernel (MAPE opt.)",
    "LS-RMSPE + Sym. Kernel (RMSPE opt.)",
    "ε-SVR (MSE)",
    "Random Forest",
    "XGBoost",
    "Linear Regression",
    "WLS (1/y²)",
    "Log-SVR",
    "QR (τ = 0.5)"
  ),
  abbrev = c(
    "SVR-MAPE", "SVR-MAPE+SK",
    "LS-RMSPE-M", "LS-RMSPE-R",
    "LS-RMSPE+SK-M", "LS-RMSPE+SK-R",
    "SVR-MSE", "RF", "XGB", "LR", "WLS", "Log-SVR", "QR"
  ),
  family = c(rep("psvr", 6), rep("baseline", 7))
)

# ── SECTION 2: Metric functions ───────────────────────────────────────────────

# Mean Absolute Percentage Error
mape_fn <- function(y, yhat) {
  mean(abs((y - yhat) / y)) * 100
}

# Root Mean Square Percentage Error
rmspe_fn <- function(y, yhat) {
  sqrt(mean(((y - yhat) / y)^2)) * 100
}

# Mean Arctangent Absolute Percentage Error (Kim & Kim 2016)
# Scaled to [0, 100] via division by pi/2 * 100
maape_fn <- function(y, yhat) {
  mean(atan(abs((y - yhat) / y))) * (200 / pi)
}

# Mean Absolute Scaled Error
# For cross-sectional data, m = 1 (lag-1 naive denominator).
# The denominator depends on row order; this is documented in each article.
mase_fn <- function(y, yhat, y_train, m = 1L) {
  denom <- mean(abs(diff(y_train, lag = m)))
  mean(abs(y - yhat)) / denom
}

# Mean Squared Error
mse_fn <- function(y, yhat) {
  mean((y - yhat)^2)
}

# Coefficient of determination
r2_fn <- function(y, yhat) {
  1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
}

# Compute all metrics at once; returns a one-row tibble
compute_metrics <- function(y, yhat, y_train) {
  tibble::tibble(
    MAPE  = mape_fn(y, yhat),
    RMSPE = rmspe_fn(y, yhat),
    MAAPE = maape_fn(y, yhat),
    MASE  = mase_fn(y, yhat, y_train),
    MSE   = mse_fn(y, yhat),
    R2    = r2_fn(y, yhat)
  )
}

# ── SECTION 3: 5-fold CV grid search ─────────────────────────────────────────

# Returns the row of `grid` that minimises mean CV metric_fn across k folds.
cv_grid <- function(X_tr, y_tr, fit_fn, pred_fn, grid, metric_fn,
                    k = 5L, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n    <- nrow(X_tr)
  fold <- sample(rep(seq_len(k), length.out = n))

  scores <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    fold_scores <- numeric(k)
    for (j in seq_len(k)) {
      val_idx  <- which(fold == j)
      Xf_tr    <- X_tr[-val_idx, , drop = FALSE]
      yf_tr    <- y_tr[-val_idx]
      Xf_val   <- X_tr[ val_idx, , drop = FALSE]
      yf_val   <- y_tr[ val_idx]
      fit      <- tryCatch(
        fit_fn(Xf_tr, yf_tr, grid[i, ]),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        fold_scores[j] <- Inf
      } else {
        pred           <- tryCatch(pred_fn(fit, Xf_val), error = function(e) NULL)
        fold_scores[j] <- if (is.null(pred)) Inf else metric_fn(yf_val, pred)
      }
    }
    scores[i] <- mean(fold_scores, na.rm = TRUE)
  }
  grid[which.min(scores), ]
}

# ── SECTION 4: Model definitions ─────────────────────────────────────────────

# Returns a named list of 13 model objects.
# Each model is a list with:
#   $label     : character — display name (matches MODEL_LABELS$label)
#   $abbrev    : character — short name (matches MODEL_LABELS$abbrev)
#   $family    : character — "psvr" or "baseline"
#   $fit_fn    : function(X_tr, y_tr, params) -> fit object
#   $pred_fn   : function(fit, X_new) -> numeric vector
#   $grid      : data.frame of hyperparameter combinations
#   $cv_metric : function(y, yhat) -> scalar (CV minimisation target)
#
# p = number of predictors (integer); used to set mtry grid for RF.

make_models <- function(p) {

  # Shared hyperparameter grids
  rbf_sigmas <- c(0.01, 0.1, 1, 10)
  Cs         <- c(0.1, 1, 10, 100)
  epsilons   <- c(0.01, 0.1, 1)
  Gammas     <- c(0.1, 1, 10, 100)

  models <- list()

  # ── m1: SVR-MAPE ────────────────────────────────────────────────────────────
  models$m1 <- list(
    label     = "SVR-MAPE",
    abbrev    = "SVR-MAPE",
    family    = "psvr",
    fit_fn    = function(X, y, params) {
      K <- psvr::make_kernel("rbf", sigma = params$sigma)
      psvr::mape_svr(X, y, kernel = K, C = params$C, eps = params$eps)
    },
    pred_fn   = function(fit, Xn) predict(fit, Xn),
    grid      = expand.grid(C = Cs, eps = epsilons, sigma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = mape_fn
  )

  # ── m2: SVR-MAPE + Sym. Kernel ──────────────────────────────────────────────
  models$m2 <- list(
    label     = "SVR-MAPE + Sym. Kernel",
    abbrev    = "SVR-MAPE+SK",
    family    = "psvr",
    fit_fn    = function(X, y, params) {
      K <- psvr::make_kernel("rbf", sigma = params$sigma)
      psvr::mape_sym_svr(X, y, kernel = K, C = params$C, eps = params$eps,
                         a = 1L)
    },
    pred_fn   = function(fit, Xn) predict(fit, Xn),
    grid      = expand.grid(C = Cs, eps = epsilons, sigma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = mape_fn
  )

  # ── m3a: LS-RMSPE (MAPE opt.) ───────────────────────────────────────────────
  models$m3a <- list(
    label     = "LS-RMSPE (MAPE opt.)",
    abbrev    = "LS-RMSPE-M",
    family    = "psvr",
    fit_fn    = function(X, y, params) {
      K <- psvr::make_kernel("rbf", sigma = params$sigma)
      psvr::rmspe_lssvr(X, y, kernel = K, gamma = params$Gamma)
    },
    pred_fn   = function(fit, Xn) predict(fit, Xn),
    grid      = expand.grid(Gamma = Gammas, sigma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = mape_fn
  )

  # ── m3b: LS-RMSPE (RMSPE opt.) ──────────────────────────────────────────────
  models$m3b <- list(
    label     = "LS-RMSPE (RMSPE opt.)",
    abbrev    = "LS-RMSPE-R",
    family    = "psvr",
    fit_fn    = function(X, y, params) {
      K <- psvr::make_kernel("rbf", sigma = params$sigma)
      psvr::rmspe_lssvr(X, y, kernel = K, gamma = params$Gamma)
    },
    pred_fn   = function(fit, Xn) predict(fit, Xn),
    grid      = expand.grid(Gamma = Gammas, sigma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = rmspe_fn
  )

  # ── m4a: LS-RMSPE + Sym. Kernel (MAPE opt.) ─────────────────────────────────
  models$m4a <- list(
    label     = "LS-RMSPE + Sym. Kernel (MAPE opt.)",
    abbrev    = "LS-RMSPE+SK-M",
    family    = "psvr",
    fit_fn    = function(X, y, params) {
      K <- psvr::make_kernel("rbf", sigma = params$sigma)
      psvr::rmspe_sym_lssvr(X, y, kernel = K, gamma = params$Gamma, a = 1L)
    },
    pred_fn   = function(fit, Xn) predict(fit, Xn),
    grid      = expand.grid(Gamma = Gammas, sigma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = mape_fn
  )

  # ── m4b: LS-RMSPE + Sym. Kernel (RMSPE opt.) ────────────────────────────────
  models$m4b <- list(
    label     = "LS-RMSPE + Sym. Kernel (RMSPE opt.)",
    abbrev    = "LS-RMSPE+SK-R",
    family    = "psvr",
    fit_fn    = function(X, y, params) {
      K <- psvr::make_kernel("rbf", sigma = params$sigma)
      psvr::rmspe_sym_lssvr(X, y, kernel = K, gamma = params$Gamma, a = 1L)
    },
    pred_fn   = function(fit, Xn) predict(fit, Xn),
    grid      = expand.grid(Gamma = Gammas, sigma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = rmspe_fn
  )

  # ── b1: ε-SVR (MSE) ─────────────────────────────────────────────────────────
  models$b1 <- list(
    label     = "ε-SVR (MSE)",
    abbrev    = "SVR-MSE",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      e1071::svm(X, y, type = "eps-regression", kernel = "radial",
                 cost = params$cost, epsilon = params$eps,
                 gamma = params$gamma, scale = FALSE)
    },
    pred_fn   = function(fit, Xn) as.numeric(predict(fit, Xn)),
    grid      = expand.grid(cost = Cs, eps = epsilons, gamma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    cv_metric = function(y, yhat) sqrt(mean((y - yhat)^2))
  )

  # ── b2: Random Forest ───────────────────────────────────────────────────────
  mtry_vals <- unique(c(2L, floor(sqrt(p)), floor(p / 2L)))
  models$b2 <- list(
    label     = "Random Forest",
    abbrev    = "RF",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      ranger::ranger(y = y, x = as.data.frame(X),
                     num.trees = 500L, mtry = params$mtry, seed = 1L)
    },
    pred_fn   = function(fit, Xn) {
      predict(fit, data = as.data.frame(Xn))$predictions
    },
    grid      = data.frame(mtry = mtry_vals),
    cv_metric = function(y, yhat) sqrt(mean((y - yhat)^2))
  )

  # ── b3: XGBoost ─────────────────────────────────────────────────────────────
  models$b3 <- list(
    label     = "XGBoost",
    abbrev    = "XGB",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      dtrain <- xgboost::xgb.DMatrix(data = X, label = y)
      xgboost::xgboost(
        data      = dtrain,
        nrounds   = params$nrounds,
        eta       = params$eta,
        max_depth = params$max_depth,
        objective = "reg:squarederror",
        verbose   = 0
      )
    },
    pred_fn   = function(fit, Xn) {
      xgboost::predict(fit, xgboost::xgb.DMatrix(data = Xn))
    },
    grid      = expand.grid(
      nrounds   = c(100L, 300L),
      eta       = c(0.05, 0.1),
      max_depth = c(3L, 6L),
      stringsAsFactors = FALSE
    ),
    cv_metric = function(y, yhat) sqrt(mean((y - yhat)^2))
  )

  # ── b4: Linear Regression ───────────────────────────────────────────────────
  models$b4 <- list(
    label     = "Linear Regression",
    abbrev    = "LR",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      lm(y ~ ., data = as.data.frame(X))
    },
    pred_fn   = function(fit, Xn) {
      as.numeric(predict(fit, newdata = as.data.frame(Xn)))
    },
    grid      = data.frame(dummy = 1L),
    cv_metric = function(y, yhat) sqrt(mean((y - yhat)^2))
  )

  # ── b5: WLS (1/y²) ──────────────────────────────────────────────────────────
  models$b5 <- list(
    label     = "WLS (1/y²)",
    abbrev    = "WLS",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      lm(y ~ ., data = as.data.frame(X), weights = 1 / y^2)
    },
    pred_fn   = function(fit, Xn) {
      as.numeric(predict(fit, newdata = as.data.frame(Xn)))
    },
    grid      = data.frame(dummy = 1L),
    cv_metric = function(y, yhat) sqrt(mean((y - yhat)^2))
  )

  # ── b6: Log-SVR ─────────────────────────────────────────────────────────────
  # Classical epsilon-SVR fitted on log(y); predictions exponentiated.
  # CV metric evaluated on the original scale to avoid circularity.
  models$b6 <- list(
    label     = "Log-SVR",
    abbrev    = "Log-SVR",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      e1071::svm(X, log(y), type = "eps-regression", kernel = "radial",
                 cost = params$cost, epsilon = params$eps,
                 gamma = params$gamma, scale = FALSE)
    },
    pred_fn   = function(fit, Xn) {
      as.numeric(exp(predict(fit, Xn)))
    },
    grid      = expand.grid(cost = Cs, eps = epsilons, gamma = rbf_sigmas,
                            stringsAsFactors = FALSE),
    # CV metric on original scale (MAPE of exp(pred) vs y)
    cv_metric = mape_fn
  )

  # ── b7a: QR (τ = 0.5) ───────────────────────────────────────────────────────
  models$b7a <- list(
    label     = "QR (τ = 0.5)",
    abbrev    = "QR",
    family    = "baseline",
    fit_fn    = function(X, y, params) {
      quantreg::rq(y ~ ., data = as.data.frame(X), tau = 0.5)
    },
    pred_fn   = function(fit, Xn) {
      as.numeric(predict(fit, newdata = as.data.frame(Xn)))
    },
    grid      = data.frame(dummy = 1L),
    cv_metric = function(y, yhat) sqrt(mean((y - yhat)^2))
  )

  models
}

# ── SECTION 5: Experiment runner ─────────────────────────────────────────────

# Runs the full 30-seed protocol on a single dataset.
# Returns a tidy data frame with columns:
#   dataset | seed | model_id | label | abbrev | family |
#   MAPE | RMSPE | MAAPE | MASE | MSE | R2
#
# Arguments:
#   X            : numeric matrix of predictors (already scaled if needed)
#   y            : numeric vector of strictly positive targets
#   dataset_name : character label for the dataset column
#   seeds        : integer vector of random seeds (default 1:30)
#   verbose      : print progress to console

run_experiment <- function(X, y, dataset_name, seeds = 1:30,
                           verbose = TRUE) {
  stopifnot(is.matrix(X), is.numeric(y), all(y > 0))
  p      <- ncol(X)
  models <- make_models(p)

  results <- vector("list", length(seeds) * length(models))
  idx     <- 1L

  for (s in seeds) {
    set.seed(s)
    n      <- nrow(X)
    tr_idx <- sample(n, floor(0.8 * n))
    X_tr   <- X[ tr_idx, , drop = FALSE]
    y_tr   <- y[ tr_idx]
    X_te   <- X[-tr_idx, , drop = FALSE]
    y_te   <- y[-tr_idx]

    for (mid in names(models)) {
      if (verbose)
        message(sprintf("[%s] seed %02d/%d  %s",
                        dataset_name, s, max(seeds), mid))

      m <- models[[mid]]

      yhat <- tryCatch({
        best_params <- cv_grid(
          X_tr      = X_tr,
          y_tr      = y_tr,
          fit_fn    = m$fit_fn,
          pred_fn   = m$pred_fn,
          grid      = m$grid,
          metric_fn = m$cv_metric,
          k         = 5L,
          seed      = s
        )
        fit <- m$fit_fn(X_tr, y_tr, best_params)
        m$pred_fn(fit, X_te)
      }, error = function(e) {
        warning(sprintf("[%s] seed %d model %s failed: %s",
                        dataset_name, s, mid, conditionMessage(e)))
        rep(NA_real_, length(y_te))
      })

      met <- if (anyNA(yhat)) {
        tibble::tibble(MAPE = NA_real_, RMSPE = NA_real_, MAAPE = NA_real_,
                       MASE = NA_real_, MSE   = NA_real_, R2    = NA_real_)
      } else {
        compute_metrics(y_te, yhat, y_tr)
      }

      results[[idx]] <- dplyr::bind_cols(
        tibble::tibble(
          dataset  = dataset_name,
          seed     = s,
          model_id = mid,
          label    = m$label,
          abbrev   = m$abbrev,
          family   = m$family
        ),
        met
      )
      idx <- idx + 1L
    }
  }

  dplyr::bind_rows(results)
}

# ── SECTION 6: Summary and statistical helpers ────────────────────────────────

# Percentile bootstrap CI for a numeric vector.
# Returns named vector: lower, mean, upper.
bootstrap_ci <- function(x, B = 1000L, alpha = 0.05, seed = 42L) {
  set.seed(seed)
  x    <- x[!is.na(x)]
  if (length(x) == 0L)
    return(c(lower = NA_real_, mean = NA_real_, upper = NA_real_))
  boot <- replicate(B, mean(sample(x, replace = TRUE)))
  c(
    lower = unname(quantile(boot, alpha / 2)),
    mean  = mean(x),
    upper = unname(quantile(boot, 1 - alpha / 2))
  )
}

# Tidy summary: mean + 95% bootstrap CI per model per metric.
# Returns one row per model, sorted ascending by MAPE_mean.
summarise_results <- function(results,
                              metrics = c("MAPE","RMSPE","MAAPE",
                                          "MASE","MSE","R2")) {
  results |>
    dplyr::group_by(model_id, label, abbrev, family) |>
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(metrics),
        list(
          mean  = \(x) mean(x, na.rm = TRUE),
          lower = \(x) bootstrap_ci(x)[["lower"]],
          upper = \(x) bootstrap_ci(x)[["upper"]]
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    ) |>
    dplyr::arrange(MAPE_mean)
}

# Paired Wilcoxon signed-rank test of each model vs. the best baseline.
# "Best baseline" = baseline model with lowest mean MAPE across seeds.
# Returns a data frame with columns:
#   model_id | label | reference | p_value | signif
wilcoxon_vs_best <- function(results, metric = "MAPE") {
  baseline_ids <- c("b1","b2","b3","b4","b5","b6","b7a")

  best_base <- results |>
    dplyr::filter(model_id %in% baseline_ids) |>
    dplyr::group_by(model_id) |>
    dplyr::summarise(m = mean(.data[[metric]], na.rm = TRUE),
                     .groups = "drop") |>
    dplyr::slice_min(m, n = 1L) |>
    dplyr::pull(model_id)

  ref_vec <- results |>
    dplyr::filter(model_id == best_base) |>
    dplyr::arrange(seed) |>
    dplyr::pull(dplyr::all_of(metric))

  # Pull per-model vectors via a join, avoiding cur_data()
  model_vecs <- results |>
    dplyr::filter(model_id != best_base) |>
    dplyr::arrange(model_id, seed) |>
    dplyr::select(model_id, label, abbrev, family, seed,
                  dplyr::all_of(metric)) |>
    dplyr::rename(value = dplyr::all_of(metric))

  model_vecs |>
    dplyr::group_by(model_id, label, abbrev, family) |>
    dplyr::summarise(
      p_value = tryCatch(
        stats::wilcox.test(
          value[order(seed)], ref_vec,
          paired = TRUE, exact = FALSE
        )$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      reference = best_base,
      signif    = dplyr::case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01  ~ "**",
        p_value < 0.05  ~ "*",
        TRUE            ~ "ns"
      )
    ) |>
    dplyr::arrange(p_value)
}
