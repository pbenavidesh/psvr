# Per-row prediction loop shared by predict.psvr_fit().
#
# Inputs (object fields):
#   object$loss         "mape" | "rmspe"  (branched on for coefficient choice)
#   object$sym          NULL | -1L | +1L  (selects the sym vs non-sym kernel)
#   object$kernel       closure from make_kernel()
#   object$beta         n_sv-vector of pruned dual differences  (loss = "mape")
#   object$alpha        N-vector of LS-SVR multipliers          (loss = "rmspe")
#   object$b            scalar bias
#   object$support_data n_sv x p (MAPE) or N x p (RMSPE) matrix of training inputs
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
  # MAPE prediction uses the pruned β (post-pruning); RMSPE uses the full α.
  coef_vec <- if (object$loss == "mape") object$beta else object$alpha

  for (i in seq_len(M)) {
    if (is_sym) {
      kv <- sym_kernel_vector(object$kernel, object$support_data,
                              newdata[i, ], a)
    } else {
      kv <- kernel_matrix(object$kernel, object$support_data,
                          newdata[i, , drop = FALSE])
    }
    preds[i] <- sum(coef_vec * kv) + object$b
  }
  as.numeric(preds)
}
