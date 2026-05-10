# Resolve the `precondition` argument shared by the LS-SVR fitters
# (Models 3 and 4). Returns a single logical (TRUE = apply, FALSE = don't).
#
# Accepted values:
#   "always"  → TRUE
#   "never"   → FALSE
#   "auto"    → TRUE iff max(y)/min(y) > 10
#   numeric t → TRUE iff max(y)/min(y) > t  (positive scalar)
.resolve_precondition <- function(precondition, y) {
  if (is.character(precondition)) {
    if (length(precondition) != 1L)
      stop("`precondition` must be a length-1 character or numeric scalar")
    choices <- c("always", "never", "auto")
    if (!precondition %in% choices)
      stop(sprintf(
        "`precondition` must be one of %s, or a positive numeric threshold",
        paste(sprintf('"%s"', choices), collapse = ", ")
      ))
    switch(
      precondition,
      always = TRUE,
      never  = FALSE,
      auto   = (max(y) / min(y)) > 10
    )
  } else if (is.numeric(precondition)) {
    if (length(precondition) != 1L || !is.finite(precondition) || precondition <= 0)
      stop("`precondition` numeric threshold must be a single positive finite value")
    (max(y) / min(y)) > as.numeric(precondition)
  } else {
    stop('`precondition` must be one of "always", "never", "auto", or a positive numeric threshold')
  }
}
