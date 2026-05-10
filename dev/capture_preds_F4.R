## F4 drift-quantification helper.
##
## Runs the exact 16 + 12 = 28 snapshot fits from
## tests/testthat/test-bit-identical.R and tests/testthat/test-psvr-direct.R
## and dumps a named list of prediction vectors to an RDS file. Used to
## quantify F3 -> F4 drift element-wise.
##
## Usage:
##   Rscript dev/capture_preds_F4.R <output.rds>

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1L) args[[1]] else "dev/preds_capture.rds"

suppressPackageStartupMessages({
  devtools::load_all(".")
  library(parsnip)
})

# Shared fixture --------------------------------------------------------
make_fixture <- function() {
  set.seed(2026)
  X      <- matrix(stats::rnorm(50 * 5), 50, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  y      <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
  X_test <- matrix(stats::rnorm(20 * 5), 20, 5,
                   dimnames = list(NULL, paste0("V", 1:5)))
  list(X = X, y = y, X_test = X_test, df_test = as.data.frame(X_test))
}

HP <- list(C = 10, eps = 5, gamma = 100, rbf_sigma = 1,
           degree = 2L, scale_factor = 1, a = 1L)

preds <- list()

# ---- 4 direct-fitter (deprecated wrappers) ----------------------------
fx <- make_fixture()
K_rbf <- make_kernel("rbf", sigma = HP$rbf_sigma)

preds[["bit-identical: Model 1 mape_svr (RBF) - direct golden"]] <- {
  fit <- suppressWarnings(mape_svr(fx$X, fx$y, kernel = K_rbf,
                                   C = HP$C, eps = HP$eps))
  predict(fit, fx$X_test)
}
preds[["bit-identical: Model 2 mape_sym_svr (RBF) - direct golden"]] <- {
  fit <- suppressWarnings(mape_sym_svr(fx$X, fx$y, kernel = K_rbf,
                                       C = HP$C, eps = HP$eps, a = HP$a))
  predict(fit, fx$X_test)
}
preds[["bit-identical: Model 3 rmspe_lssvr (RBF) - direct golden"]] <- {
  fit <- suppressWarnings(rmspe_lssvr(fx$X, fx$y, kernel = K_rbf,
                                      gamma = HP$gamma))
  predict(fit, fx$X_test)
}
preds[["bit-identical: Model 4 rmspe_sym_lssvr (RBF) - direct golden"]] <- {
  fit <- suppressWarnings(rmspe_sym_lssvr(fx$X, fx$y, kernel = K_rbf,
                                          gamma = HP$gamma, a = HP$a))
  predict(fit, fx$X_test)
}

# ---- 12 parsnip-pipeline tests ----------------------------------------
fit_and_predict <- function(spec, fx) {
  fit_obj <- parsnip::fit_xy(spec, x = fx$X, y = fx$y)
  predict(fit_obj, new_data = fx$df_test)$.pred
}

preds[["bit-identical: psvr_mape_rbf - parsnip golden"]] <- {
  spec <- psvr_mape_rbf(cost = HP$C, svm_margin = HP$eps,
                        rbf_sigma = HP$rbf_sigma) |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_mape_poly - parsnip golden"]] <- {
  spec <- psvr_mape_poly(cost = HP$C, svm_margin = HP$eps,
                         degree = HP$degree,
                         scale_factor = HP$scale_factor) |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_mape_linear - parsnip golden"]] <- {
  spec <- psvr_mape_linear(cost = HP$C, svm_margin = HP$eps) |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_mape_sym_rbf - parsnip golden"]] <- {
  spec <- psvr_mape_sym_rbf(cost = HP$C, svm_margin = HP$eps,
                            rbf_sigma = HP$rbf_sigma,
                            sym_type = "even") |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_mape_sym_poly - parsnip golden"]] <- {
  spec <- psvr_mape_sym_poly(cost = HP$C, svm_margin = HP$eps,
                             degree = HP$degree,
                             scale_factor = HP$scale_factor) |>
    set_engine("psvr", a = HP$a)
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_mape_sym_linear - parsnip golden"]] <- {
  spec <- psvr_mape_sym_linear(cost = HP$C, svm_margin = HP$eps) |>
    set_engine("psvr", a = HP$a)
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_rmspe_rbf - parsnip golden"]] <- {
  spec <- psvr_rmspe_rbf(cost = HP$gamma, rbf_sigma = HP$rbf_sigma) |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_rmspe_poly - parsnip golden"]] <- {
  spec <- psvr_rmspe_poly(cost = HP$gamma, degree = HP$degree,
                          scale_factor = HP$scale_factor) |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_rmspe_linear - parsnip golden"]] <- {
  spec <- psvr_rmspe_linear(cost = HP$gamma) |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_rmspe_sym_rbf - parsnip golden"]] <- {
  spec <- psvr_rmspe_sym_rbf(cost = HP$gamma,
                             rbf_sigma = HP$rbf_sigma,
                             sym_type = "even") |>
    set_engine("psvr")
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_rmspe_sym_poly - parsnip golden"]] <- {
  spec <- psvr_rmspe_sym_poly(cost = HP$gamma, degree = HP$degree,
                              scale_factor = HP$scale_factor) |>
    set_engine("psvr", a = HP$a)
  fit_and_predict(spec, fx)
}
preds[["bit-identical: psvr_rmspe_sym_linear - parsnip golden"]] <- {
  spec <- psvr_rmspe_sym_linear(cost = HP$gamma) |>
    set_engine("psvr", a = HP$a)
  fit_and_predict(spec, fx)
}

# ---- 12 direct psvr() tests -------------------------------------------
K_poly <- make_kernel("polynomial", degree = HP$degree,
                      coef0 = HP$scale_factor)
K_lin  <- make_kernel("linear")

preds[["psvr-direct: mape / no sym / RBF"]] <-
  predict(psvr(fx$X, fx$y, loss = "mape", kernel = K_rbf,
               C = HP$C, eps = HP$eps), fx$X_test)
preds[["psvr-direct: mape / no sym / poly"]] <-
  predict(psvr(fx$X, fx$y, loss = "mape", kernel = K_poly,
               C = HP$C, eps = HP$eps), fx$X_test)
preds[["psvr-direct: mape / no sym / linear"]] <-
  predict(psvr(fx$X, fx$y, loss = "mape", kernel = K_lin,
               C = HP$C, eps = HP$eps), fx$X_test)
preds[["psvr-direct: mape / sym=+1 / RBF"]] <-
  predict(psvr(fx$X, fx$y, loss = "mape", sym = HP$a, kernel = K_rbf,
               C = HP$C, eps = HP$eps), fx$X_test)
preds[["psvr-direct: mape / sym=+1 / poly"]] <-
  predict(psvr(fx$X, fx$y, loss = "mape", sym = HP$a, kernel = K_poly,
               C = HP$C, eps = HP$eps), fx$X_test)
preds[["psvr-direct: mape / sym=+1 / linear"]] <-
  predict(psvr(fx$X, fx$y, loss = "mape", sym = HP$a, kernel = K_lin,
               C = HP$C, eps = HP$eps), fx$X_test)
preds[["psvr-direct: rmspe / no sym / RBF"]] <-
  predict(psvr(fx$X, fx$y, loss = "rmspe", kernel = K_rbf,
               gamma = HP$gamma), fx$X_test)
preds[["psvr-direct: rmspe / no sym / poly"]] <-
  predict(psvr(fx$X, fx$y, loss = "rmspe", kernel = K_poly,
               gamma = HP$gamma), fx$X_test)
preds[["psvr-direct: rmspe / no sym / linear"]] <-
  predict(psvr(fx$X, fx$y, loss = "rmspe", kernel = K_lin,
               gamma = HP$gamma), fx$X_test)
preds[["psvr-direct: rmspe / sym=+1 / RBF"]] <-
  predict(psvr(fx$X, fx$y, loss = "rmspe", sym = HP$a, kernel = K_rbf,
               gamma = HP$gamma), fx$X_test)
preds[["psvr-direct: rmspe / sym=+1 / poly"]] <-
  predict(psvr(fx$X, fx$y, loss = "rmspe", sym = HP$a, kernel = K_poly,
               gamma = HP$gamma), fx$X_test)
preds[["psvr-direct: rmspe / sym=+1 / linear"]] <-
  predict(psvr(fx$X, fx$y, loss = "rmspe", sym = HP$a, kernel = K_lin,
               gamma = HP$gamma), fx$X_test)

cat(sprintf("Captured %d prediction vectors.\n", length(preds)))
saveRDS(preds, out_path)
cat(sprintf("Saved to: %s\n", out_path))
