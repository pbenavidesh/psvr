# experiment_helpers.R
# 2026-04-26
# Tidymodels-based runner for psvr cross-section case studies (Phase 3).
# Replaces tune_grid with tune_bayes (GP surrogate + EI) and uses the
# native SMO solver (psvr >= 0.3.0) as default backend.
# Phase 2 grid-search results preserved as *-results-grid-osqp.csv.
#
# Per-seed nested-resampling pipeline: outer 80/20 split via
# rsample::make_splits, inner 5-fold CV via vfold_cv, tuning via
# workflow_set + tune_bayes, final eval via last_fit.
#
# BO budget per model follows the 5×p rule (p = number of tunable HPs),
# targeting up to 50 evaluations per model with no_improve = 15 early stop:
#   m1 (3 HPs):  initial=15, iter=35
#   m2 (4 HPs):  initial=20, iter=30
#   m3 (2 HPs):  initial=10, iter=40
#   m4 (3 HPs):  initial=15, iter=35
#   b1 (2 HPs):  initial=10, iter=40
#   b2 (1 HP) :  initial= 5, iter=45
#   b3 (3 HPs):  initial=15, iter=35
#
# Parameter ranges match the Phase 2 grid bounds (cost ∈ [0.1, 100],
# svm_margin ∈ [0.01, 1], rbf_sigma data-driven via sigma_heuristic, etc.)
# so the BO comparison against grid-osqp is a fair search-strategy test
# (same search space, different sampler).
#
# Output CSV schema: downstream sections of the QMDs (summary table, box plots,
# Wilcoxon) operate on metric columns only and ignore HP columns.
#
# SCHEMA CHANGE NOTE:
# Phase 3 tune_results objects from tune_bayes have a different internal
# structure than the Phase 2 tune_grid results. Existing partial RDS files
# from Phase 2 (results/partial/*.rds) MUST be deleted before re-running,
# otherwise the resume logic will load Phase 2 results and skip BO seeds:
#
#   rm vignettes/articles/case-studies/results/partial/*.rds
#
# The Phase 2 final CSVs are preserved as *-results-grid-osqp.csv and
# committed to the repo as the grid-search baseline.
#
# Tune objects (workflow_set tune_results, one per seed) are persisted as
# results/<dataset>-tune-results.rds, mirroring electricity-forecasting.qmd's
# electricity-tune-results.rds. This enables post-hoc extraction of selected
# hyperparameters and tuning artifacts without re-running 30 seeds.
# These RDS files are ~15-20 MB per dataset; they are .gitignored.
#
# Callers must library() these packages: tidyverse, tidymodels, psvr,
# kernlab, ranger, xgboost, quantreg, e1071, hardhat, furrr, future.

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

mape_fn <- function(y, yhat) {
  mean(abs((y - yhat) / y)) * 100
}

rmspe_fn <- function(y, yhat) {
  sqrt(mean(((y - yhat) / y)^2)) * 100
}

maape_fn <- function(y, yhat) {
  mean(atan(abs((y - yhat) / y))) * (200 / pi)
}

mase_fn <- function(y, yhat, y_train, m = 1L) {
  denom <- mean(abs(diff(y_train, lag = m)))
  mean(abs(y - yhat)) / denom
}

mse_fn <- function(y, yhat) {
  mean((y - yhat)^2)
}

r2_fn <- function(y, yhat) {
  1 - sum((y - yhat)^2) / sum((y - mean(y))^2)
}

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

# Defensive: collect_predictions on a quantreg-engine workflow returns a
# .pred_quantile column (a hardhat::quantile_pred object) instead of .pred.
# For tau = 0.5 this is a single value per row; convert to numeric.
.extract_pred <- function(preds) {
  if (".pred" %in% names(preds)) {
    as.numeric(preds$.pred)
  } else if (".pred_quantile" %in% names(preds)) {
    pq <- preds$.pred_quantile
    tryCatch(
      as.numeric(pq),
      error = function(e) unlist(pq, use.names = FALSE)
    )
  } else {
    stop("No prediction column found. Names: ",
         paste(names(preds), collapse = ", "))
  }
}

# ── SECTION 3: Specs ──────────────────────────────────────────────────────────
# Returns a named list keyed by tune_id (m1, m2, m3, m4, b1, b2, b3) with
#   $spec          parsnip model_spec
#   $select_metric character vector of CV metrics to call select_best() with
#
# Phase 3: grids removed — parameter ranges are set per-workflow inside
# run_seed() via param_info / option_add(), and tune_bayes samples from
# those ranges using a GP surrogate. The function name is kept for callsite
# stability.

make_specs_and_grids <- function() {
  spec_m1 <- psvr::psvr_mape_rbf(
    cost = tune::tune(), svm_margin = tune::tune(),
    rbf_sigma = tune::tune()
  ) |>
    parsnip::set_engine("psvr")

  spec_m2 <- psvr::psvr_mape_sym_rbf(
    cost = tune::tune(), svm_margin = tune::tune(),
    rbf_sigma = tune::tune(), sym_type = tune::tune()
  ) |>
    parsnip::set_engine("psvr")

  spec_m3 <- psvr::psvr_rmspe_rbf(
    cost = tune::tune(), rbf_sigma = tune::tune()
  ) |>
    parsnip::set_engine("psvr")

  spec_m4 <- psvr::psvr_rmspe_sym_rbf(
    cost = tune::tune(), rbf_sigma = tune::tune(),
    sym_type = tune::tune()
  ) |>
    parsnip::set_engine("psvr")

  spec_b1 <- parsnip::svm_rbf(
    cost = tune::tune(), rbf_sigma = tune::tune()
  ) |>
    parsnip::set_engine("kernlab") |>
    parsnip::set_mode("regression")

  spec_b2 <- parsnip::rand_forest(
    mtry = tune::tune(), trees = 500
  ) |>
    parsnip::set_engine("ranger", seed = 1L) |>
    parsnip::set_mode("regression")

  spec_b3 <- parsnip::boost_tree(
    trees = tune::tune(), learn_rate = tune::tune(),
    tree_depth = tune::tune()
  ) |>
    parsnip::set_engine("xgboost") |>
    parsnip::set_mode("regression")

  list(
    m1 = list(spec = spec_m1, select_metric = "mape"),
    m2 = list(spec = spec_m2, select_metric = "mape"),
    m3 = list(spec = spec_m3, select_metric = c("mape", "rmse")),  # m3a, m3b
    m4 = list(spec = spec_m4, select_metric = c("mape", "rmse")),  # m4a, m4b
    b1 = list(spec = spec_b1, select_metric = "rmse"),
    b2 = list(spec = spec_b2, select_metric = "rmse"),
    b3 = list(spec = spec_b3, select_metric = "rmse")
  )
}

# ── SECTION 4: Bespoke Log-SVR seed-loop (b6) ────────────────────────────────
# Log-SVR (e1071::svm on log(y), exponentiated for prediction, tuned on
# original-scale MAPE) is not expressible as a single parsnip workflow
# because tune_grid has no native concept of "transform y, evaluate metric on
# back-transformed prediction." So we run the inner CV manually.
#
# Predictors are scaled with training-fold statistics inside this function,
# matching the recipe's step_normalize behaviour for the other models.

fit_log_svr_seed <- function(X_tr, y_tr, X_te, seed,
                             cost_vals  = c(0.1, 1, 10, 100),
                             eps_vals   = c(0.01, 0.1, 1),
                             gamma_vals = c(0.224, 0.707, 2.236, 7.071),
                             k = 5L) {
  sc <- scale(X_tr)
  X_tr_s <- as.matrix(unclass(sc))
  attr(X_tr_s, "scaled:center") <- NULL
  attr(X_tr_s, "scaled:scale")  <- NULL
  X_te_s <- scale(
    X_te,
    center = attr(sc, "scaled:center"),
    scale  = attr(sc, "scaled:scale")
  )
  attr(X_te_s, "scaled:center") <- NULL
  attr(X_te_s, "scaled:scale")  <- NULL

  grid <- expand.grid(
    cost  = cost_vals,
    eps   = eps_vals,
    gamma = gamma_vals,
    stringsAsFactors = FALSE
  )

  set.seed(seed)
  n    <- nrow(X_tr_s)
  fold <- sample(rep(seq_len(k), length.out = n))

  scores <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    fold_scores <- numeric(k)
    for (j in seq_len(k)) {
      val_idx <- which(fold == j)
      Xf_tr <- X_tr_s[-val_idx, , drop = FALSE]
      yf_tr <- y_tr[-val_idx]
      Xf_va <- X_tr_s[ val_idx, , drop = FALSE]
      yf_va <- y_tr[ val_idx]

      fit <- tryCatch(
        e1071::svm(
          Xf_tr, log(yf_tr),
          type = "eps-regression", kernel = "radial",
          cost = grid$cost[i], epsilon = grid$eps[i],
          gamma = grid$gamma[i], scale = FALSE
        ),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        fold_scores[j] <- Inf
      } else {
        yhat <- exp(as.numeric(predict(fit, Xf_va)))
        fold_scores[j] <- mape_fn(yf_va, yhat)
      }
    }
    scores[i] <- mean(fold_scores, na.rm = TRUE)
  }

  best <- grid[which.min(scores), , drop = FALSE]
  fit_final <- e1071::svm(
    X_tr_s, log(y_tr),
    type = "eps-regression", kernel = "radial",
    cost = best$cost, epsilon = best$eps,
    gamma = best$gamma, scale = FALSE
  )
  list(
    pred = exp(as.numeric(predict(fit_final, X_te_s))),
    hp   = list(cost = best$cost, epsilon = best$eps, gamma = best$gamma)
  )
}

# ── SECTION 5: Per-seed driver ────────────────────────────────────────────────
# Returns one tibble of 13 rows (one per model_id) for the given seed.

run_seed <- function(X, y, seed, dataset_name) {
  stopifnot(is.matrix(X), is.numeric(y), all(y > 0))

  pred_names <- colnames(X)
  if (is.null(pred_names)) pred_names <- paste0("V", seq_len(ncol(X)))

  all_df <- as.data.frame(X)
  names(all_df) <- pred_names
  all_df$y    <- y
  all_df$.wts <- hardhat::importance_weights(1 / y^2)

  # Outer split: preserve v1 protocol (set.seed(s); sample(n, floor(0.8n)))
  set.seed(seed)
  n_total <- nrow(all_df)
  tr_idx  <- sample(n_total, floor(0.8 * n_total))
  va_idx  <- setdiff(seq_len(n_total), tr_idx)
  split   <- rsample::make_splits(
    list(analysis = tr_idx, assessment = va_idx),
    data = all_df
  )
  train_df <- rsample::training(split)
  test_df  <- rsample::testing(split)

  # Inner 5-fold CV
  set.seed(seed + 1000L)
  folds <- rsample::vfold_cv(train_df, v = 5L)

  # Recipe: normalize numeric predictors using training stats only.
  # train_df$.wts is an importance_weights object (hardhat); recipes
  # auto-detects the class and assigns role "case_weights", so
  # all_numeric_predictors() does not select it. Non-b5 workflows
  # leave the column unused; b5 picks it up via add_case_weights(.wts)
  # on the workflow.
  rec_default <- recipes::recipe(y ~ ., data = train_df) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  sg <- make_specs_and_grids()

  metric_set_all <- yardstick::metric_set(
    yardstick::mape, yardstick::rmse, yardstick::rsq
  )

  # ── Build tunable workflow_set ──
  wf_set <- workflowsets::workflow_set(
    preproc = list(default = rec_default),
    models  = list(
      m1 = sg$m1$spec, m2 = sg$m2$spec,
      m3 = sg$m3$spec, m4 = sg$m4$spec,
      b1 = sg$b1$spec, b2 = sg$b2$spec, b3 = sg$b3$spec
    )
  )

  train_baked    <- rec_default |>
    recipes::prep() |>
    recipes::bake(new_data = train_df)
  predictor_only <- train_baked |> dplyr::select(-y, -.wts)

  # ── Per-workflow param_info (matching Phase 2 grid bounds for fair
  #    BO-vs-grid comparison; rbf_sigma data-driven for psvr models) ──
  rbf_sigma_param   <- psvr::rbf_sigma_psvr_data(predictor_only)
  cost_param        <- dials::cost(
    range = c(-1, 2), trans = scales::log10_trans()
  )
  svm_margin_param  <- dials::svm_margin(
    range = c(-2, 0), trans = scales::log10_trans()
  )
  rbf_sigma_kernlab <- dials::rbf_sigma(
    range = c(log10(0.266), log10(1.494)),
    trans = scales::log10_trans()
  )
  mtry_param_b2     <- dials::mtry(
    range = c(2L, max(2L, floor(ncol(X) / 2L)))
  )
  trees_param       <- dials::trees(range = c(100L, 300L))
  learn_rate_param  <- dials::learn_rate(range = c(0.05, 0.1), trans = NULL)
  tree_depth_param  <- dials::tree_depth(range = c(3L, 6L))

  set_pi <- function(wf_set, wf_id, ...) {
    pi <- workflowsets::extract_workflow(wf_set, id = wf_id) |>
      tune::extract_parameter_set_dials() |>
      stats::update(...)
    workflowsets::option_add(wf_set, param_info = pi, id = wf_id)
  }
  wf_set <- set_pi(wf_set, "default_m1",
                   cost = cost_param, svm_margin = svm_margin_param,
                   rbf_sigma = rbf_sigma_param)
  wf_set <- set_pi(wf_set, "default_m2",
                   cost = cost_param, svm_margin = svm_margin_param,
                   rbf_sigma = rbf_sigma_param)
  wf_set <- set_pi(wf_set, "default_m3",
                   cost = cost_param, rbf_sigma = rbf_sigma_param)
  wf_set <- set_pi(wf_set, "default_m4",
                   cost = cost_param, rbf_sigma = rbf_sigma_param)
  wf_set <- set_pi(wf_set, "default_b1",
                   cost = cost_param, rbf_sigma = rbf_sigma_kernlab)
  wf_set <- set_pi(wf_set, "default_b2",
                   mtry = mtry_param_b2)
  wf_set <- set_pi(wf_set, "default_b3",
                   trees = trees_param, learn_rate = learn_rate_param,
                   tree_depth = tree_depth_param)

  # ── Tune each model via Bayesian optimisation (GP + EI) ──
  bayes_iters <- list(
    m1 = list(initial = 15L, iter = 35L),
    m2 = list(initial = 20L, iter = 30L),
    m3 = list(initial = 10L, iter = 40L),
    m4 = list(initial = 15L, iter = 35L),
    b1 = list(initial = 10L, iter = 40L),
    b2 = list(initial =  5L, iter = 45L),
    b3 = list(initial = 15L, iter = 35L)
  )

  ctrl_bayes <- tune::control_bayes(
    verbose    = FALSE,
    no_improve = 15L,
    save_pred  = FALSE,
    allow_par  = FALSE,
    seed       = seed
  )

  tune_results <- list()
  for (mid in c("m1", "m2", "m3", "m4", "b1", "b2", "b3")) {
    wf_id <- paste0("default_", mid)
    tune_results[[mid]] <- wf_set |>
      dplyr::filter(wflow_id == wf_id) |>
      workflowsets::workflow_map(
        fn        = "tune_bayes",
        resamples = folds,
        initial   = bayes_iters[[mid]]$initial,
        iter      = bayes_iters[[mid]]$iter,
        metrics   = metric_set_all,
        control   = ctrl_bayes,
        seed      = seed,
        verbose   = FALSE
      )
  }

  # ── Score helper: finalize tuned workflow, last_fit, compute metrics ──
  score_tuned <- function(mid, model_id, label, abbrev, family,
                          select_metric) {
    wf_id <- paste0("default_", mid)
    res <- workflowsets::extract_workflow_set_result(
      tune_results[[mid]], id = wf_id
    )
    wf  <- workflowsets::extract_workflow(wf_set, id = wf_id)
    best <- tune::select_best(res, metric = select_metric)
    wf_final <- tune::finalize_workflow(wf, best)
    fit_lf <- tune::last_fit(wf_final, split, metrics = metric_set_all)
    preds  <- tune::collect_predictions(fit_lf)
    yhat   <- .extract_pred(preds)
    yact   <- preds$y
    met    <- compute_metrics(yact, yhat, y_train = train_df$y)
    tibble::tibble(
      dataset  = dataset_name, seed = seed,
      model_id = model_id, label = label,
      abbrev   = abbrev, family = family
    ) |>
      dplyr::bind_cols(met) |>
      dplyr::mutate(
        cost_selected      = if ("cost"       %in% names(best)) best$cost       else NA_real_,
        epsilon_selected   = if ("svm_margin" %in% names(best)) best$svm_margin else NA_real_,
        sigma_selected     = if ("rbf_sigma"  %in% names(best)) best$rbf_sigma  else NA_real_,
        gamma_selected     = if ("rbf_sigma"  %in% names(best)) 1 / (2 * best$rbf_sigma^2) else NA_real_,
        sym_type_selected  = if ("sym_type"   %in% names(best)) as.character(best$sym_type) else NA_character_,
        mtry_selected      = if ("mtry"       %in% names(best)) best$mtry       else NA_integer_,
        xgb_trees_selected = if ("trees"      %in% names(best)) best$trees      else NA_integer_,
        xgb_lr_selected    = if ("learn_rate" %in% names(best)) best$learn_rate else NA_real_,
        xgb_depth_selected = if ("tree_depth" %in% names(best)) best$tree_depth else NA_integer_
      )
  }

  score_untuned <- function(wf, model_id, label, abbrev, family) {
    fit_lf <- tune::last_fit(wf, split, metrics = metric_set_all)
    preds  <- tune::collect_predictions(fit_lf)
    yhat   <- .extract_pred(preds)
    yact   <- preds$y
    met    <- compute_metrics(yact, yhat, y_train = train_df$y)
    tibble::tibble(
      dataset  = dataset_name, seed = seed,
      model_id = model_id, label = label,
      abbrev   = abbrev, family = family
    ) |>
      dplyr::bind_cols(met) |>
      dplyr::mutate(
        cost_selected      = NA_real_,
        epsilon_selected   = NA_real_,
        sigma_selected     = NA_real_,
        gamma_selected     = NA_real_,
        sym_type_selected  = NA_character_,
        mtry_selected      = NA_integer_,
        xgb_trees_selected = NA_integer_,
        xgb_lr_selected    = NA_real_,
        xgb_depth_selected = NA_integer_
      )
  }

  results <- list()

  results$m1  <- score_tuned("m1",  "m1",  "SVR-MAPE",
                             "SVR-MAPE",      "psvr", "mape")
  results$m2  <- score_tuned("m2",  "m2",  "SVR-MAPE + Sym. Kernel",
                             "SVR-MAPE+SK",   "psvr", "mape")
  results$m3a <- score_tuned("m3",  "m3a", "LS-RMSPE (MAPE opt.)",
                             "LS-RMSPE-M",    "psvr", "mape")
  results$m3b <- score_tuned("m3",  "m3b", "LS-RMSPE (RMSPE opt.)",
                             "LS-RMSPE-R",    "psvr", "rmse")
  results$m4a <- score_tuned("m4",  "m4a", "LS-RMSPE + Sym. Kernel (MAPE opt.)",
                             "LS-RMSPE+SK-M", "psvr", "mape")
  results$m4b <- score_tuned("m4",  "m4b", "LS-RMSPE + Sym. Kernel (RMSPE opt.)",
                             "LS-RMSPE+SK-R", "psvr", "rmse")
  results$b1  <- score_tuned("b1",  "b1",  "ε-SVR (MSE)",
                             "SVR-MSE",       "baseline", "rmse")
  results$b2  <- score_tuned("b2",  "b2",  "Random Forest",
                             "RF",            "baseline", "rmse")
  results$b3  <- score_tuned("b3",  "b3",  "XGBoost",
                             "XGB",           "baseline", "rmse")

  # ── Untuned: b4 (lm) ───────────────────────────────────────────
  spec_b4 <- parsnip::linear_reg() |> parsnip::set_engine("lm")
  wf_b4 <- workflows::workflow() |>
    workflows::add_recipe(rec_default) |>
    workflows::add_model(spec_b4)
  results$b4 <- score_untuned(wf_b4, "b4", "Linear Regression",
                              "LR", "baseline")

  # ── Untuned: b5 (WLS via case_weights) ─────────────────────────
  spec_b5 <- parsnip::linear_reg() |> parsnip::set_engine("lm")
  wf_b5 <- workflows::workflow() |>
    workflows::add_case_weights(.wts) |>
    workflows::add_recipe(rec_default) |>
    workflows::add_model(spec_b5)
  results$b5 <- score_untuned(wf_b5, "b5", "WLS (1/y²)",
                              "WLS", "baseline")

  # ── b6: bespoke Log-SVR ────────────────────────────────────────
  b6_out <- tryCatch(
    fit_log_svr_seed(
      X_tr = X[tr_idx, , drop = FALSE], y_tr = y[tr_idx],
      X_te = X[va_idx, , drop = FALSE], seed = seed
    ),
    error = function(e) {
      warning(sprintf("[%s] seed %d b6 failed: %s",
                      dataset_name, seed, conditionMessage(e)))
      list(pred = rep(NA_real_, length(va_idx)), hp = NULL)
    }
  )
  yhat_b6 <- b6_out$pred
  b6_hp   <- b6_out$hp
  met_b6 <- if (anyNA(yhat_b6)) {
    tibble::tibble(MAPE = NA_real_, RMSPE = NA_real_, MAAPE = NA_real_,
                   MASE = NA_real_, MSE = NA_real_, R2 = NA_real_)
  } else {
    compute_metrics(y[va_idx], yhat_b6, y_train = y[tr_idx])
  }
  results$b6 <- tibble::tibble(
    dataset  = dataset_name, seed = seed,
    model_id = "b6", label = "Log-SVR",
    abbrev   = "Log-SVR", family = "baseline"
  ) |>
    dplyr::bind_cols(met_b6) |>
    dplyr::mutate(
      cost_selected      = if (!is.null(b6_hp)) b6_hp$cost    else NA_real_,
      epsilon_selected   = if (!is.null(b6_hp)) b6_hp$epsilon else NA_real_,
      sigma_selected     = NA_real_,
      gamma_selected     = if (!is.null(b6_hp)) b6_hp$gamma   else NA_real_,
      sym_type_selected  = NA_character_,
      mtry_selected      = NA_integer_,
      xgb_trees_selected = NA_integer_,
      xgb_lr_selected    = NA_real_,
      xgb_depth_selected = NA_integer_
    )

  # ── b7a: bespoke quantreg::rq with τ = 0.5 ─────────────────────
  # parsnip's "quantile regression" mode returns quantile_pred objects
  # that are not compatible with the regression yardstick metrics
  # (mape/rmse/rsq) used in metric_set_all, so last_fit() would fail.
  # Bypass parsnip and call quantreg::rq() directly. The .wts column
  # is dropped from the baked data so it is not pulled in as a
  # predictor by the y ~ . formula.
  yhat_b7a <- tryCatch({
    rec_prepped <- rec_default |> recipes::prep(training = train_df)
    train_baked_b7a <- rec_prepped |>
      recipes::bake(new_data = train_df) |>
      dplyr::select(-dplyr::any_of(".wts"))
    test_baked_b7a  <- rec_prepped |>
      recipes::bake(new_data = test_df) |>
      dplyr::select(-dplyr::any_of(".wts"))
    fit_b7a <- quantreg::rq(y ~ ., data = train_baked_b7a, tau = 0.5)
    as.numeric(predict(fit_b7a, newdata = test_baked_b7a))
  }, error = function(e) {
    warning(sprintf("[%s] seed %d b7a failed: %s",
                    dataset_name, seed, conditionMessage(e)))
    rep(NA_real_, nrow(test_df))
  })
  met_b7a <- if (anyNA(yhat_b7a)) {
    tibble::tibble(MAPE = NA_real_, RMSPE = NA_real_, MAAPE = NA_real_,
                   MASE = NA_real_, MSE = NA_real_, R2 = NA_real_)
  } else {
    compute_metrics(test_df$y, yhat_b7a, y_train = train_df$y)
  }
  results$b7a <- tibble::tibble(
    dataset  = dataset_name, seed = seed,
    model_id = "b7a", label = "QR (τ = 0.5)",
    abbrev   = "QR", family = "baseline"
  ) |>
    dplyr::bind_cols(met_b7a) |>
    dplyr::mutate(
      cost_selected      = NA_real_,
      epsilon_selected   = NA_real_,
      sigma_selected     = NA_real_,
      gamma_selected     = NA_real_,
      sym_type_selected  = NA_character_,
      mtry_selected      = NA_integer_,
      xgb_trees_selected = NA_integer_,
      xgb_lr_selected    = NA_real_,
      xgb_depth_selected = NA_integer_
    )

  list(
    metrics = dplyr::bind_rows(results),
    tune    = tune_results
  )
}

# ── SECTION 6: Sequential experiment runner ───────────────────────────────────
# In-process driver for QMD render-time fallback (when CSV is missing).
# Wraps run_seed() over `seeds` and returns a single tidy tibble (metrics only;
# tune objects are not persisted in sequential mode).

run_experiment <- function(X, y, dataset_name, seeds = 1:30, verbose = TRUE) {
  stopifnot(is.matrix(X), is.numeric(y), all(y > 0))

  results <- vector("list", length(seeds))
  for (i in seq_along(seeds)) {
    s <- seeds[i]
    if (verbose) {
      message(sprintf("[%s] seed %02d/%d", dataset_name, s, max(seeds)))
    }
    results[[i]] <- tryCatch(
      run_seed(X, y, seed = s, dataset_name = dataset_name)$metrics,
      error = function(e) {
        warning(sprintf("[%s] seed %d failed: %s",
                        dataset_name, s, conditionMessage(e)))
        NULL
      }
    )
  }
  dplyr::bind_rows(purrr::compact(results))
}

# ── SECTION 7: Parallel runner with per-seed RDS partials ─────────────────────
# Per-seed partials saved under results/partial/<dataset>_seed_NN.rds; the run
# is resumable. Parallelisation is over seeds — each worker runs run_seed()
# sequentially (no nested inner parallelism).

run_experiment_parallel <- function(X, y, dataset_name,
                                    seeds   = 1:30,
                                    workers = NULL) {
  if (is.null(workers)) {
    workers <- max(1L, parallel::detectCores() - 1L)
  }

  partial_dir <- file.path("results", "partial")
  dir.create(partial_dir, showWarnings = FALSE, recursive = TRUE)

  done <- seeds[file.exists(
    file.path(partial_dir,
              sprintf("%s_seed_%02d.rds", dataset_name, seeds))
  )]
  todo <- setdiff(seeds, done)

  if (length(done) > 0L) {
    message(sprintf(
      "[%s] Skipping %d completed seeds: %s",
      dataset_name, length(done), paste(done, collapse = " ")))
  }

  if (length(todo) > 0L) {
    message(sprintf(
      "[%s] Running %d seeds on %d workers...",
      dataset_name, length(todo),
      min(workers, length(todo))))

    future::plan(future::multisession, workers = min(workers, length(todo)))
    t0 <- proc.time()

    furrr::future_walk(todo, function(s) {
      # Worker-side dependencies — recipes/workflows/tune are loaded via
      # tidymodels; engine packages must be available for set_engine().
      require(tidymodels, quietly = TRUE)
      require(psvr,       quietly = TRUE)
      require(kernlab,    quietly = TRUE)
      require(ranger,     quietly = TRUE)
      require(xgboost,    quietly = TRUE)
      require(quantreg,   quietly = TRUE)
      require(e1071,      quietly = TRUE)

      seed_results <- tryCatch(
        run_seed(X, y, seed = s, dataset_name = dataset_name),
        error = function(e) {
          warning(sprintf("[%s] seed %d failed: %s",
                          dataset_name, s, conditionMessage(e)))
          NULL
        }
      )

      if (!is.null(seed_results)) {
        saveRDS(
          seed_results$metrics,
          file.path("results", "partial",
                    sprintf("%s_seed_%02d.rds", dataset_name, s))
        )
        saveRDS(
          seed_results$tune,
          file.path("results", "partial",
                    sprintf("%s_seed_%02d_tune.rds", dataset_name, s))
        )
      }
    }, .options = furrr::furrr_options(seed = TRUE))

    future::plan(future::sequential)
    elapsed <- round((proc.time() - t0)[["elapsed"]] / 60, 1)
    message(sprintf("[%s] Seeds done in %.1f min.",
                    dataset_name, elapsed))
  }

  rds_files <- file.path(
    partial_dir,
    sprintf("%s_seed_%02d.rds", dataset_name, seeds))
  missing <- rds_files[!file.exists(rds_files)]
  if (length(missing) > 0L) {
    stop(sprintf("[%s] Missing partials: %s",
                 dataset_name,
                 paste(basename(missing), collapse = ", ")))
  }

  results <- purrr::map_dfr(rds_files, readRDS)
  out_csv <- file.path(
    "results",
    sprintf("%s-results.csv", gsub("_", "-", dataset_name)))
  readr::write_csv(results, out_csv)
  message(sprintf(
    "[%s] Saved: %s  (%d rows, %d NA-MAPE)\n",
    dataset_name, out_csv,
    nrow(results), sum(is.na(results$MAPE))))

  tune_rds_files <- file.path(
    partial_dir,
    sprintf("%s_seed_%02d_tune.rds", dataset_name, seeds))
  tune_present <- tune_rds_files[file.exists(tune_rds_files)]
  if (length(tune_present) == length(seeds)) {
    tune_consolidated <- purrr::map(
      seeds,
      function(s) {
        readRDS(file.path(partial_dir,
                          sprintf("%s_seed_%02d_tune.rds", dataset_name, s)))
      }
    )
    names(tune_consolidated) <- sprintf("seed_%02d", seeds)
    out_tune_rds <- file.path(
      "results",
      sprintf("%s-tune-results.rds", gsub("_", "-", dataset_name)))
    saveRDS(tune_consolidated, out_tune_rds)
    message(sprintf("[%s] Saved consolidated tune objects: %s",
                    dataset_name, out_tune_rds))
  } else {
    message(sprintf(
      "[%s] Skipping tune consolidation: %d/%d tune partials present.",
      dataset_name, length(tune_present), length(seeds)))
  }

  invisible(results)
}

# ── SECTION 8: Summary and statistical helpers ────────────────────────────────
# Unchanged from v1 — they operate on the CSV schema.

bootstrap_ci <- function(x, B = 1000L, alpha = 0.05, seed = 42L) {
  set.seed(seed)
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(c(lower = NA_real_, mean = NA_real_, upper = NA_real_))
  }
  boot <- replicate(B, mean(sample(x, replace = TRUE)))
  c(
    lower = unname(quantile(boot, alpha / 2)),
    mean  = mean(x),
    upper = unname(quantile(boot, 1 - alpha / 2))
  )
}

summarise_results <- function(results,
                              metrics = c("MAPE", "RMSPE", "MAAPE",
                                          "MASE", "MSE", "R2")) {
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

wilcoxon_vs_best <- function(results, metric = "MAPE") {
  baseline_ids <- c("b1", "b2", "b3", "b4", "b5", "b6", "b7a")

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
