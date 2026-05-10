# Per-row prediction loop shared by predict.psvr_fit().
#
# Inputs (object fields):
#   object$loss         "mape" | "rmspe"  (informational; not branched on here)
#   object$sym          NULL | -1L | +1L  (selects the sym vs non-sym kernel)
#   object$kernel       closure from make_kernel()
#   object$alpha        N-vector of dual coefficients (β for ε-SVR; α for LS-SVR)
#   object$b            scalar bias
#   object$support_data N x p matrix of support vectors (or full X for LS-SVR)
#   object$p_train      training feature count, used for column validation
#
# Returns a plain numeric vector of length nrow(newdata).
.psvr_predict_dispatch <- function(object, newdata) {
  newdata <- as.matrix(newdata)
  p <- ncol(newdata)
  if (p != object$p_train)
    stop(sprintf("newdata has %d column%s but model was trained on %d",
                 p, if (p == 1L) "" else "s", object$p_train))

  M     <- nrow(newdata)
  preds <- numeric(M)
  is_sym <- !is.null(object$sym)
  a      <- if (is_sym) as.integer(object$sym) else NA_integer_

  for (i in seq_len(M)) {
    if (is_sym) {
      kv <- sym_kernel_vector(object$kernel, object$support_data,
                              newdata[i, ], a)
    } else {
      kv <- kernel_matrix(object$kernel, object$support_data,
                          newdata[i, , drop = FALSE])
    }
    preds[i] <- sum(object$alpha * kv) + object$b
  }
  as.numeric(preds)
}
