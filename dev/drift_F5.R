## Compare F4 vs F5 predictions and emit a drift + iter table.
##
## Reads dev/preds_F4.rds, dev/preds_F5.rds, dev/iters_F4.rds, dev/iters_F5.rds
## and prints one row per snapshot test with the following columns:
##   F4 iter | F4 conv | F5 iter | F5 conv | Max abs diff | Verdict
##
## Verdict categorisation:
##   F5 FIX      -> F4 stalled (max_iter, !converged) but F5 converged.
##   F5 REGRESS  -> F4 converged but F5 stalled.
##   STALE F4+F5 -> Both stalled at max_iter; "drift" is between two
##                  non-converged states (neither is correct).
##   LINSOLVE    -> RMSPE test; LS-SVR linear system, not SMO. Always 0 drift.
##   LEGIT       -> Both converged; drift within ~tol*max(y).
##   BUG?        -> Both converged but drift >> 10 * tol*max(y).

f4 <- readRDS("dev/preds_F4.rds")
f5 <- readRDS("dev/preds_F5.rds")
iters_f4 <- readRDS("dev/iters_F4.rds")
iters_f5 <- readRDS("dev/iters_F5.rds")

stopifnot(identical(names(f4), names(f5)))

# Snapshot fixture max(y) for the "expected drift" scaling.
set.seed(2026)
y_fix <- stats::rlnorm(50, meanlog = 0, sdlog = 0.5)
y_max <- max(y_fix)
tol   <- 1e-3
expected_per_pred <- tol * y_max

cat(sprintf("Snapshot fixture: max(y) = %.4f, tol = %.0e\n", y_max, tol))
cat(sprintf("Expected per-pair drift order: ~%.2e\n", expected_per_pred))
cat(sprintf("Red-flag threshold (10x):       %.2e\n\n", 10 * expected_per_pred))

classify_test <- function(name) {
  if (grepl("rmspe", name, ignore.case = TRUE)) return(NULL)
  has_sym <- grepl("_sym_|sym=\\+1|Model 2", name)
  is_rbf  <- grepl("rbf|RBF", name)
  is_poly <- grepl("poly", name)
  is_lin  <- grepl("linear", name)
  if (has_sym) {
    if (is_rbf)  return("mape sym rbf")
    if (is_poly) return("mape sym poly")
    if (is_lin)  return("mape sym lin")
  } else {
    if (is_rbf)  return("mape rbf")
    if (is_poly) return("mape poly")
    if (is_lin)  return("mape lin")
  }
  NULL
}

to_num <- function(x) {
  if (is.list(x) && "predictions" %in% names(x)) return(as.numeric(x$predictions))
  if (is.data.frame(x)) return(as.numeric(x[[1]]))
  as.numeric(x)
}

verdict <- function(c4, c5, max_diff, smo_case) {
  if (is.null(smo_case)) return("LINSOLVE")
  if (!c4 && !c5) return("STALE F4+F5")
  if (!c4 &&  c5) return("F5 FIX")
  if ( c4 && !c5) return("F5 REGRESS")
  if (max_diff > 10 * expected_per_pred) return("BUG?")
  "LEGIT"
}

rows <- vector("list", length(f4))
for (i in seq_along(f4)) {
  v4 <- to_num(f4[[i]])
  v5 <- to_num(f5[[i]])
  d  <- abs(v5 - v4)
  smo_case <- classify_test(names(f4)[[i]])
  if (is.null(smo_case)) {
    f4_iter <- NA_integer_; f4_conv <- NA
    f5_iter <- NA_integer_; f5_conv <- NA
  } else {
    f4_iter <- iters_f4[[smo_case]]$iterations
    f4_conv <- iters_f4[[smo_case]]$converged
    f5_iter <- iters_f5[[smo_case]]$iterations
    f5_conv <- iters_f5[[smo_case]]$converged
  }
  rows[[i]] <- data.frame(
    test     = names(f4)[[i]],
    f4_iter  = f4_iter, f4_conv = f4_conv,
    f5_iter  = f5_iter, f5_conv = f5_conv,
    max_diff = max(d),
    verdict  = verdict(f4_conv, f5_conv, max(d), smo_case),
    stringsAsFactors = FALSE
  )
}
df <- do.call(rbind, rows)
df <- df[order(factor(df$verdict,
                      levels = c("BUG?", "F5 REGRESS", "STALE F4+F5",
                                 "F5 FIX", "LEGIT", "LINSOLVE")),
              -df$max_diff), ]

fmt_iter <- function(x) if (is.na(x)) "  -   " else sprintf("%6d", x)
fmt_conv <- function(x) if (is.na(x)) "  -  " else if (x) " YES " else " NO  "

cat("| Test name | F4 iter | F4 conv | F5 iter | F5 conv | Max abs diff | Verdict |\n")
cat("|---|---|---|---|---|---|---|\n")
for (i in seq_len(nrow(df))) {
  cat(sprintf("| %s | %s | %s | %s | %s | %.3e | %s |\n",
              df$test[i],
              fmt_iter(df$f4_iter[i]),
              fmt_conv(df$f4_conv[i]),
              fmt_iter(df$f5_iter[i]),
              fmt_conv(df$f5_conv[i]),
              df$max_diff[i],
              df$verdict[i]))
}

cat("\nVerdict counts:\n")
print(table(df$verdict))
cat("\nKey numbers:\n")
cat(sprintf("  Total tests:       %d\n", nrow(df)))
cat(sprintf("  BUG? (red flag):   %d\n", sum(df$verdict == "BUG?")))
cat(sprintf("  F5 REGRESS:        %d\n", sum(df$verdict == "F5 REGRESS")))
cat(sprintf("  Max drift overall: %.3e\n", max(df$max_diff)))

ok <- !is.na(df$f4_iter) & df$f4_conv & df$f5_conv
if (any(ok)) {
  cat("\nIter-count comparison (both converged):\n")
  for (i in which(ok)) {
    cat(sprintf("  %-55s F4: %5d  F5: %5d  (%+d, %.1f%%)\n",
                df$test[i], df$f4_iter[i], df$f5_iter[i],
                df$f5_iter[i] - df$f4_iter[i],
                100 * (df$f5_iter[i] - df$f4_iter[i]) / df$f4_iter[i]))
  }
}
