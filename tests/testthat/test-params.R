test_that("margin_percentage() returns a dials param with range [1, 20]", {
  p <- margin_percentage()
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, 1)
  expect_equal(r$upper, 20)
})

test_that("margin_percentage() accepts a custom range", {
  p <- margin_percentage(range = c(0.5, 10))
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, 0.5)
  expect_equal(r$upper, 10)
})

test_that("sigma_heuristic() returns a positive scalar on a small matrix", {
  set.seed(42)
  X <- matrix(rnorm(200), ncol = 4)
  result <- sigma_heuristic(X)
  expect_length(result, 1L)
  expect_true(is.numeric(result))
  expect_gt(result, 0)
})

test_that("sigma_heuristic() subsamples when nrow(X) > sample_size", {
  set.seed(1)
  X <- matrix(rnorm(2000), ncol = 4)   # 500 rows
  result_full    <- sigma_heuristic(X, sample_size = 500L, seed = 7L)
  result_sampled <- sigma_heuristic(X, sample_size = 20L,  seed = 7L)
  # Both are positive scalars; the subsampled result differs from the full one
  expect_gt(result_full,    0)
  expect_gt(result_sampled, 0)
  expect_false(isTRUE(all.equal(result_full, result_sampled)))
})

test_that("sigma_heuristic() respects the seed argument", {
  X <- matrix(rnorm(2000), ncol = 4)
  r1 <- sigma_heuristic(X, sample_size = 20L, seed = 99L)
  r2 <- sigma_heuristic(X, sample_size = 20L, seed = 99L)
  r3 <- sigma_heuristic(X, sample_size = 20L, seed = 42L)
  expect_equal(r1, r2)
  expect_false(isTRUE(all.equal(r1, r3)))
})

test_that("rbf_sigma_psvr() returns a dials param with default range [-3, 1]", {
  p <- rbf_sigma_psvr()
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, -3)
  expect_equal(r$upper, 1)
})

test_that("rbf_sigma_psvr() has NULL finalize", {
  p <- rbf_sigma_psvr()
  expect_null(p$finalize)
})

test_that("rbf_sigma_psvr_data() returns a quant_param with data-driven range", {
  set.seed(42)
  X <- matrix(rnorm(200), ncol = 4)
  p <- rbf_sigma_psvr_data(X)
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_true(is.finite(r$lower))
  expect_true(is.finite(r$upper))
  expect_lt(r$lower, r$upper)
})

test_that("rbf_sigma_psvr_data() width argument changes range width", {
  set.seed(42)
  X <- matrix(rnorm(200), ncol = 4)
  p1 <- rbf_sigma_psvr_data(X, width = 10)
  p2 <- rbf_sigma_psvr_data(X, width = 100)
  r1 <- dials::range_get(p1, original = FALSE)
  r2 <- dials::range_get(p2, original = FALSE)
  expect_lt(r1$upper - r1$lower, r2$upper - r2$lower)
})

test_that("psvr_option_add() updates rbf_sigma for psvr workflows only", {
  skip_if_not_installed("workflowsets")
  skip_if_not_installed("tune")
  skip_if_not_installed("recipes")
  skip_if_not_installed("parsnip")

  rec <- recipes::recipe(mpg ~ ., data = mtcars) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  spec_m3 <- psvr_rmspe_rbf(cost = tune::tune(), rbf_sigma = tune::tune()) |>
    parsnip::set_engine("psvr")
  spec_lm <- parsnip::linear_reg() |> parsnip::set_engine("lm")

  wf_set <- workflowsets::workflow_set(
    preproc = list(base = rec),
    models  = list(m3_rmspe = spec_m3, lm = spec_lm)
  )

  X_baked <- rec |> recipes::prep() |>
    recipes::bake(new_data = mtcars) |>
    dplyr::select(-mpg)

  wf_set2 <- psvr_option_add(wf_set, X_baked, seed = 42L)

  # psvr workflow should have options set; lm workflow should not
  m3_opts <- workflowsets::extract_workflow_set_result  # check via wflow_id
  psvr_wf_id <- wf_set2$wflow_id[grepl("m3|m1|m2|m4", wf_set2$wflow_id)]
  lm_wf_id   <- wf_set2$wflow_id[!grepl("m3|m1|m2|m4", wf_set2$wflow_id)]

  # psvr workflow has options; lm does not
  psvr_opts <- wf_set2$option[[which(wf_set2$wflow_id == psvr_wf_id)]]
  lm_opts   <- wf_set2$option[[which(wf_set2$wflow_id == lm_wf_id)]]
  expect_true(length(psvr_opts) > 0)
  expect_true(length(lm_opts)   == 0)
})

test_that("cost_psvr() returns a dials param with range [-2, 10]", {
  p <- cost_psvr()
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, -2)
  expect_equal(r$upper, 10)
})

test_that("cost_psvr_ls_data() returns a quant_param with data-driven upper bound", {
  set.seed(42)
  y <- abs(rnorm(100, mean = 50, sd = 10)) + 1
  p <- cost_psvr_ls_data(y)
  expect_s3_class(p, "quant_param")
  r <- dials::range_get(p, original = FALSE)
  expect_equal(r$lower, -2)
  # Upper bound is at least log2(var(y) * N) (width_log2 = 4 only widens it)
  expect_gte(r$upper, log2(var(y) * length(y)))
})

test_that("cost_psvr_ls_data() upper bound covers Boston Housing Gamma_opt", {
  skip_if_not_installed("mlbench")
  data("BostonHousing", package = "mlbench")
  y_full <- BostonHousing$medv
  # 80/20 split used by run_seed(): N_train = floor(0.8 * 506) = 404
  p <- cost_psvr_ls_data(y_full, n = 404L)
  r <- dials::range_get(p, original = FALSE)
  # Published Boston optimum: Gamma ~= 16962.54  ->  log2 ~= 14.05
  expect_gte(r$upper, log2(16962.54))
})

test_that("cost_psvr_ls_data() rejects non-positive y or n", {
  expect_error(cost_psvr_ls_data(c(-1, 2, 3)))
  expect_error(cost_psvr_ls_data(c(0, 1, 2)))
  expect_error(cost_psvr_ls_data(c(1, 2, 3), n = 0))
})

test_that("cost_psvr_ls_data() warns on negative width_log2", {
  y <- c(10, 20, 30, 40, 50)
  expect_warning(
    cost_psvr_ls_data(y, width_log2 = -2),
    "width_log2 < 0"
  )
})

test_that("psvr_option_add_cost_ls() updates cost for m3/m4 workflows only", {
  skip_if_not_installed("workflowsets")
  skip_if_not_installed("tune")
  skip_if_not_installed("recipes")
  skip_if_not_installed("parsnip")

  rec <- recipes::recipe(mpg ~ ., data = mtcars) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  spec_m3 <- psvr_rmspe_rbf(cost = tune::tune(), rbf_sigma = tune::tune()) |>
    parsnip::set_engine("psvr")
  spec_m1 <- psvr_mape_rbf(cost = tune::tune(),
                            svm_margin = tune::tune(),
                            rbf_sigma  = tune::tune()) |>
    parsnip::set_engine("psvr")

  wf_set <- workflowsets::workflow_set(
    preproc = list(base = rec),
    models  = list(m3_rmspe = spec_m3, m1_mape = spec_m1)
  )

  wf_set2 <- psvr_option_add_cost_ls(wf_set, y = mtcars$mpg)

  m3_id <- wf_set2$wflow_id[grepl("m3", wf_set2$wflow_id)]
  m1_id <- wf_set2$wflow_id[grepl("m1", wf_set2$wflow_id)]

  m3_opts <- wf_set2$option[[which(wf_set2$wflow_id == m3_id)]]
  m1_opts <- wf_set2$option[[which(wf_set2$wflow_id == m1_id)]]
  # m3 has options set; m1 does not
  expect_true(length(m3_opts) > 0)
  expect_true(length(m1_opts) == 0)
})
