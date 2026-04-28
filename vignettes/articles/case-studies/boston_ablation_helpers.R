# boston_ablation_helpers.R
# Cell runners and per-seed driver for the Boston Housing 2x2 reflection-
# augmentation ablation. Sourced by boston-reflection-ablation.qmd.
#
# Design (see paper Section 4.4 + task spec 2026-04-28):
#   * 8 cells = {ε-SVR-MAPE, LS-SVR-RMSPE} × {non-sym, sym a=+1} ×
#               {non-augmented, augmented}
#   * Standardization order: standardize FIRST on the non-augmented training
#     fold, augment SECOND in standardized space. This keeps μ/σ identical
#     across all 8 cells; only the {Z, -Z} duplication differs.
#   * For non-augmented cells (A1, A3, B1, B3): tune_bayes via single-model
#     parsnip workflow. Standard 5-fold CV — no leakage concern.
#   * For augmented cells (A2, A4, B2, B4): hand-rolled 50-point Latin
#     hypercube + paired 5-fold CV. Folds are constructed on the original N
#     rows; augmentation happens INSIDE each fold's training portion only.
#     This avoids the leakage that naive 5-fold over 2N augmented rows would
#     introduce (mirrored pair split across folds).
#
# Methodological caveat (Phase 1 confirmation): LHS-50 is not strictly
# equivalent to tune_bayes(initial=15, iter=35) at the same total budget.
# tune_bayes typically converges to better optima at small budgets via the
# GP+EI surrogate. This may bias Δ_aug *against* augmented cells (worse
# tuning). If Δ_sym still dominates Δ_aug under this anti-symmetric bias,
# the conclusion is robust.

# ── Augmentation ─────────────────────────────────────────────────────────
augment_reflect <- function(X, y) {
  list(X = rbind(X, -X), y = c(y, y))
}

.assert_no_test_leak <- function(X_train_used, X_test) {
  stk <- rbind(X_train_used, X_test)
  dup <- duplicated(stk)[seq.int(nrow(X_train_used) + 1L, nrow(stk))]
  stopifnot("test row appears in (possibly reflected) training fold" = !any(dup))
}

# ── Standardization (denominator N-1, matches recipes::step_normalize) ──
.std_train <- function(X) {
  mu <- colMeans(X)
  s  <- apply(X, 2, sd)
  s[s == 0 | is.na(s)] <- 1
  list(mu = mu, sigma = s, X_std = sweep(sweep(X, 2, mu, "-"), 2, s, "/"))
}
.std_apply <- function(X, mu, s) sweep(sweep(X, 2, mu, "-"), 2, s, "/")

# ── Latin hypercube in [0, 1)^k (no extra dependency) ────────────────────
.lhs_unit <- function(n, k, seed) {
  set.seed(seed)
  m <- matrix(NA_real_, n, k)
  for (j in seq_len(k)) {
    m[, j] <- (sample.int(n) - 1 + runif(n)) / n
  }
  m
}

# Map LHS draws to actual hyperparameter values. cost is log2,
# rbf_sigma is log10, svm_margin is linear (% of target).
.lhs_hp_grid <- function(family, n,
                          cost_range_log2,
                          sigma_range_log10,
                          margin_range = c(1, 20),
                          seed) {
  k <- if (family == "A") 3L else 2L
  u <- .lhs_unit(n, k, seed)
  cost_v  <- 2  ^ (u[, 1L] * (cost_range_log2[2]   - cost_range_log2[1])   + cost_range_log2[1])
  sigma_v <- 10 ^ (u[, 2L] * (sigma_range_log10[2] - sigma_range_log10[1]) + sigma_range_log10[1])
  if (family == "A") {
    margin_v <- u[, 3L] * (margin_range[2] - margin_range[1]) + margin_range[1]
    data.frame(cost = cost_v, rbf_sigma = sigma_v, svm_margin = margin_v)
  } else {
    data.frame(cost = cost_v, rbf_sigma = sigma_v)
  }
}

# ── Direct psvr fit dispatch ─────────────────────────────────────────────
.fit_psvr_cell <- function(family, sym, X_std, y, hp) {
  K <- psvr::make_kernel("rbf", sigma = hp$rbf_sigma)
  if (family == "A") {
    if (sym) {
      psvr::mape_sym_svr(X_std, y, kernel = K,
                          C = hp$cost, eps = hp$svm_margin, a = 1L)
    } else {
      psvr::mape_svr(X_std, y, kernel = K,
                      C = hp$cost, eps = hp$svm_margin)
    }
  } else {  # family B
    if (sym) {
      psvr::rmspe_sym_lssvr(X_std, y, kernel = K,
                             gamma = hp$cost, a = 1L)
    } else {
      psvr::rmspe_lssvr(X_std, y, kernel = K, gamma = hp$cost)
    }
  }
}

# ── Cell runner: paired 5-fold + LHS BO (augmented cells A2/A4/B2/B4) ───
# X_tr, y_tr: original (non-augmented) 80% training fold (length N).
# X_te, y_te: original held-out test fold (length ~0.2N).
# hp_grid:    LHS grid (50 rows × 2 or 3 cols) — passed in from caller so
#             A2 and A4 share the same grid (and similarly B2/B4) within
#             a seed. See run_seed_ablation for construction.
# fold_id:    integer vector of length N assigning each original row to
#             one of 5 folds — also passed in from caller for paired
#             coordination across A2/A4/B2/B4 within a seed.
run_cell_paired <- function(family, sym,
                             X_tr, y_tr, X_te, y_te,
                             hp_grid, fold_id) {
  stopifnot(
    "family must be 'A' or 'B'"   = family %in% c("A", "B"),
    "sym must be a single logical"= is.logical(sym) && length(sym) == 1L,
    is.matrix(X_tr), is.matrix(X_te),
    "hp_grid must be passed (not constructed locally)" = !missing(hp_grid),
    "fold_id must be passed (not constructed locally)" = !missing(fold_id),
    "fold_id length must equal nrow(X_tr)" = length(fold_id) == nrow(X_tr)
  )

  # ── Score one HP point: 5-fold mean MAPE, paired (no leakage) ──
  evaluate_hp <- function(j) {
    hp <- hp_grid[j, , drop = FALSE]
    fold_mapes <- numeric(5L)
    for (k in seq_len(5L)) {
      i_tr <- which(fold_id != k)
      i_va <- which(fold_id == k)

      # Standardize FIRST on non-augmented inner-training fold.
      std      <- .std_train(X_tr[i_tr, , drop = FALSE])
      X_va_std <- .std_apply(X_tr[i_va, , drop = FALSE], std$mu, std$sigma)

      # Augment SECOND in standardized space: rbind(Z, -Z), y duplicated.
      aug <- augment_reflect(std$X_std, y_tr[i_tr])

      fit <- tryCatch(
        .fit_psvr_cell(family, sym, aug$X, aug$y, hp),
        error = function(e) NULL
      )
      fold_mapes[k] <- if (is.null(fit)) {
        Inf
      } else {
        yhat <- as.numeric(stats::predict(fit, X_va_std))
        mean(abs((y_tr[i_va] - yhat) / y_tr[i_va])) * 100
      }
    }
    mean(fold_mapes)
  }

  # ── Evaluate all 50 LHS points; pick lowest mean inner-CV MAPE ──
  hp_mapes <- vapply(seq_len(nrow(hp_grid)), evaluate_hp, numeric(1L))

  if (all(is.infinite(hp_mapes))) {
    warning(sprintf("[%s%s, augmented] all %d LHS HP points failed CV — refit will fail",
                    family, if (sym) "_sym" else "", nrow(hp_grid)))
  }

  best_idx <- which.min(hp_mapes)
  best_hp  <- hp_grid[best_idx, , drop = FALSE]

  # ── Refit on FULL augmented training (size 2N), predict on test ──
  # Same standardize-first-then-augment order as inner CV.
  std_full <- .std_train(X_tr)
  X_te_std <- .std_apply(X_te, std_full$mu, std_full$sigma)
  aug_full <- augment_reflect(std_full$X_std, y_tr)

  .assert_no_test_leak(aug_full$X, X_te_std)

  fit_final <- tryCatch(
    .fit_psvr_cell(family, sym, aug_full$X, aug_full$y, best_hp),
    error = function(e) {
      warning(sprintf("[%s%s, augmented] final refit failed: %s",
                      family, if (sym) "_sym" else "",
                      conditionMessage(e)))
      NULL
    }
  )
  yhat_te <- if (is.null(fit_final)) rep(NA_real_, nrow(X_te))
             else as.numeric(stats::predict(fit_final, X_te_std))

  list(best_hp = best_hp, yhat = yhat_te,
       cv_mape_min = hp_mapes[best_idx],
       n_failed_lhs = sum(is.infinite(hp_mapes)))
}

# ── Cell runner: tune_bayes over single workflow (cells A1/A3/B1/B3) ─────
# Uses recipes::step_normalize → standardize on non-augmented training
# (per fold inside CV; full training fold for the final refit). No
# augmentation involved → no leakage concern.
run_cell_standard <- function(family, sym,
                               X_tr, y_tr, X_te, y_te,
                               cost_param, sigma_param, margin_param,
                               seed) {
  stopifnot(
    family %in% c("A", "B"),
    is.logical(sym), length(sym) == 1L,
    is.matrix(X_tr), is.matrix(X_te)
  )

  pred_names <- colnames(X_tr)
  if (is.null(pred_names)) pred_names <- paste0("V", seq_len(ncol(X_tr)))

  train_df <- as.data.frame(X_tr); names(train_df) <- pred_names
  train_df$y <- y_tr
  test_df  <- as.data.frame(X_te); names(test_df) <- pred_names
  test_df$y  <- y_te

  rec <- recipes::recipe(y ~ ., data = train_df) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  spec <- if (family == "A") {
    if (sym) {
      psvr::psvr_mape_sym_rbf(
        cost = tune::tune(), svm_margin = tune::tune(),
        rbf_sigma = tune::tune(), sym_type = "even"
      ) |> parsnip::set_engine("psvr")
    } else {
      psvr::psvr_mape_rbf(
        cost = tune::tune(), svm_margin = tune::tune(),
        rbf_sigma = tune::tune()
      ) |> parsnip::set_engine("psvr")
    }
  } else {
    if (sym) {
      psvr::psvr_rmspe_sym_rbf(
        cost = tune::tune(), rbf_sigma = tune::tune(),
        sym_type = "even"
      ) |> parsnip::set_engine("psvr")
    } else {
      psvr::psvr_rmspe_rbf(
        cost = tune::tune(), rbf_sigma = tune::tune()
      ) |> parsnip::set_engine("psvr")
    }
  }

  wf <- workflows::workflow() |>
    workflows::add_recipe(rec) |>
    workflows::add_model(spec)

  set.seed(seed + 1000L)
  folds <- rsample::vfold_cv(train_df, v = 5L)

  pi <- tune::extract_parameter_set_dials(wf)
  pi <- if (family == "A") {
    stats::update(pi,
                  cost       = cost_param,
                  rbf_sigma  = sigma_param,
                  svm_margin = margin_param)
  } else {
    stats::update(pi,
                  cost      = cost_param,
                  rbf_sigma = sigma_param)
  }

  metric_set_all <- yardstick::metric_set(
    yardstick::mape, yardstick::rmse, yardstick::rsq
  )
  ctrl <- tune::control_bayes(
    verbose    = FALSE,
    no_improve = 15L,
    save_pred  = FALSE,
    allow_par  = FALSE,
    seed       = seed
  )
  bo_initial <- if (family == "A") 15L else 10L
  bo_iter    <- if (family == "A") 35L else 40L

  res <- tryCatch(
    tune::tune_bayes(
      wf, resamples = folds, param_info = pi,
      initial = bo_initial, iter = bo_iter,
      metrics = metric_set_all, control = ctrl
    ),
    error = function(e) {
      warning(sprintf("[%s%s, non-augmented] tune_bayes failed: %s",
                      family, if (sym) "_sym" else "",
                      conditionMessage(e)))
      NULL
    }
  )
  if (is.null(res)) {
    return(list(best_hp = data.frame(),
                yhat = rep(NA_real_, nrow(test_df)),
                cv_mape_min = NA_real_, n_failed_lhs = NA_integer_))
  }

  best <- tune::select_best(res, metric = "mape")
  wf_final <- tune::finalize_workflow(wf, best)
  fit_obj  <- tryCatch(parsnip::fit(wf_final, data = train_df),
                       error = function(e) NULL)
  yhat <- if (is.null(fit_obj)) {
    rep(NA_real_, nrow(test_df))
  } else {
    as.numeric(stats::predict(fit_obj, new_data = test_df)$.pred)
  }
  list(best_hp = as.data.frame(best),
       yhat = yhat,
       cv_mape_min = NA_real_, n_failed_lhs = NA_integer_)
}

# ── Per-seed driver ──────────────────────────────────────────────────────
# Outer 80/20 split (mirrors run_seed convention), then runs the 8 cells
# sequentially on that split. Returns an 8-row tibble.
run_seed_ablation <- function(X, y, seed,
                               dataset_name = "boston_ablation",
                               n_lhs = 50L) {
  stopifnot(is.matrix(X), is.numeric(y), all(y > 0))
  N <- nrow(X)

  # Outer split: identical convention to experiment_helpers.R run_seed().
  set.seed(seed)
  tr_idx <- sample(N, floor(0.8 * N))
  te_idx <- setdiff(seq_len(N), tr_idx)
  X_tr <- X[tr_idx, , drop = FALSE]; y_tr <- y[tr_idx]
  X_te <- X[te_idx, , drop = FALSE]; y_te <- y[te_idx]

  # ── Search ranges (identical across all cells of a family) ──
  # rbf_sigma: from non-augmented standardized X_tr, shared by ALL 8 cells.
  std_orig <- .std_train(X_tr)
  sigma_param <- psvr::rbf_sigma_psvr_data(std_orig$X_std)
  sr <- dials::range_get(sigma_param, original = FALSE)
  sigma_range_log10 <- c(sr$lower, sr$upper)

  # cost: A static, B data-driven on original y_tr (length N, NOT 2N).
  cost_param_eps <- psvr::cost_psvr()
  cost_range_eps <- c(-2, 10)
  cost_param_ls  <- psvr::cost_psvr_ls_data(y_tr)
  cr_ls <- dials::range_get(cost_param_ls, original = FALSE)
  cost_range_ls  <- c(cr_ls$lower, cr_ls$upper)

  margin_param <- psvr::margin_percentage()

  # ── Shared LHS grids and fold_id for the 4 augmented cells ──
  # A2 and A4 receive the SAME hp_grid_A (within a seed); B2 and B4 the
  # same hp_grid_B. Same seed + 5000L → deterministic LHS per family.
  hp_grid_A <- .lhs_hp_grid(
    family = "A", n = n_lhs,
    cost_range_log2 = cost_range_eps,
    sigma_range_log10 = sigma_range_log10,
    seed = seed + 5000L
  )
  hp_grid_B <- .lhs_hp_grid(
    family = "B", n = n_lhs,
    cost_range_log2 = cost_range_ls,
    sigma_range_log10 = sigma_range_log10,
    seed = seed + 5000L
  )
  set.seed(seed + 6000L)
  fold_id <- sample(rep(seq_len(5L), length.out = nrow(X_tr)))

  # ── 8 cells ──
  cells <- tibble::tribble(
    ~cell_id, ~family, ~sym,  ~augmented,
    "A1",     "A",     FALSE, FALSE,
    "A2",     "A",     FALSE, TRUE,
    "A3",     "A",     TRUE,  FALSE,
    "A4",     "A",     TRUE,  TRUE,
    "B1",     "B",     FALSE, FALSE,
    "B2",     "B",     FALSE, TRUE,
    "B3",     "B",     TRUE,  FALSE,
    "B4",     "B",     TRUE,  TRUE
  )

  rows <- vector("list", nrow(cells))
  for (i in seq_len(nrow(cells))) {
    cl <- cells[i, ]
    t0 <- proc.time()
    r <- if (cl$augmented) {
      hp_grid <- if (cl$family == "A") hp_grid_A else hp_grid_B
      run_cell_paired(
        family = cl$family, sym = cl$sym,
        X_tr = X_tr, y_tr = y_tr,
        X_te = X_te, y_te = y_te,
        hp_grid = hp_grid, fold_id = fold_id
      )
    } else {
      cost_param <- if (cl$family == "A") cost_param_eps else cost_param_ls
      run_cell_standard(
        family = cl$family, sym = cl$sym,
        X_tr = X_tr, y_tr = y_tr,
        X_te = X_te, y_te = y_te,
        cost_param = cost_param,
        sigma_param = sigma_param,
        margin_param = margin_param,
        seed = seed
      )
    }
    elapsed_sec <- (proc.time() - t0)[["elapsed"]]

    metrics <- compute_metrics(y_te, r$yhat, y_train = y_tr)
    rows[[i]] <- tibble::tibble(
      dataset       = dataset_name,
      seed          = seed,
      cell_id       = cl$cell_id,
      family        = cl$family,
      sym           = cl$sym,
      augmented     = cl$augmented,
      cost_selected   = if ("cost"       %in% names(r$best_hp)) r$best_hp$cost       else NA_real_,
      sigma_selected  = if ("rbf_sigma"  %in% names(r$best_hp)) r$best_hp$rbf_sigma  else NA_real_,
      eps_selected    = if ("svm_margin" %in% names(r$best_hp)) r$best_hp$svm_margin else NA_real_,
      cv_mape_min     = r$cv_mape_min %||% NA_real_,
      n_failed_lhs    = r$n_failed_lhs %||% NA_integer_,
      elapsed_sec     = elapsed_sec
    ) |>
      dplyr::bind_cols(metrics)
  }
  dplyr::bind_rows(rows)
}

# Tiny helper: %||% from rlang (avoid hard dep).
`%||%` <- function(x, y) if (is.null(x) || is.na(x)) y else x

# ── Parallel runner (mirrors run_experiment_parallel pattern) ────────────
run_experiment_ablation_parallel <- function(X, y,
                                              seeds   = 1:30,
                                              workers = 12L,
                                              dataset_name = "boston_ablation") {
  partial_dir <- here::here("vignettes/articles/case-studies/results", "partial")
  dir.create(partial_dir, showWarnings = FALSE, recursive = TRUE)

  done <- seeds[file.exists(
    file.path(partial_dir,
              sprintf("%s_seed_%02d.rds", dataset_name, seeds))
  )]
  todo <- setdiff(seeds, done)

  if (length(done) > 0L) {
    message(sprintf("[%s] Skipping %d completed seeds: %s",
                    dataset_name, length(done), paste(done, collapse = " ")))
  }

  if (length(todo) > 0L) {
    message(sprintf("[%s] Running %d seeds on %d workers...",
                    dataset_name, length(todo),
                    min(workers, length(todo))))
    future::plan(future::multisession, workers = min(workers, length(todo)))
    t0 <- proc.time()

    furrr::future_walk(todo, function(s) {
      suppressPackageStartupMessages({
        library(tidymodels)
        library(psvr)
      })
      source(here::here("vignettes/articles/case-studies/experiment_helpers.R"))
      source(here::here("vignettes/articles/case-studies/boston_ablation_helpers.R"))

      seed_results <- tryCatch(
        run_seed_ablation(X, y, seed = s, dataset_name = dataset_name),
        error = function(e) {
          warning(sprintf("[%s] seed %d failed: %s",
                          dataset_name, s, conditionMessage(e)))
          NULL
        }
      )
      if (!is.null(seed_results)) {
        saveRDS(seed_results,
                file.path(partial_dir,
                          sprintf("%s_seed_%02d.rds", dataset_name, s)))
      }
    }, .options = furrr::furrr_options(seed = TRUE))

    future::plan(future::sequential)
    elapsed <- round((proc.time() - t0)[["elapsed"]] / 60, 1)
    message(sprintf("[%s] Seeds done in %.1f min.", dataset_name, elapsed))
  }

  rds_files <- file.path(
    partial_dir,
    sprintf("%s_seed_%02d.rds", dataset_name, seeds))
  missing <- rds_files[!file.exists(rds_files)]
  if (length(missing) > 0L) {
    stop(sprintf("[%s] Missing partials: %s",
                 dataset_name, paste(basename(missing), collapse = ", ")))
  }

  results <- purrr::map_dfr(rds_files, readRDS)
  out_csv <- here::here(
    "vignettes/articles/case-studies/results",
    "boston-reflection-ablation-results.csv")
  readr::write_csv(results, out_csv)
  message(sprintf("[%s] Saved: %s (%d rows)\n",
                  dataset_name, out_csv, nrow(results)))

  invisible(results)
}
