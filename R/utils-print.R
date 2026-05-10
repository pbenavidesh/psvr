# Internal helper: build a human-readable kernel description from kernel_info.
# Used by the four print.psvr_*() methods on the OLD classes and by
# print.psvr_fit() on the unified class.
.kernel_desc <- function(ki) {
  if (is.null(ki)) return("user-supplied function")
  switch(ki$type,
    rbf        = sprintf("RBF (sigma = %g)", ki$sigma),
    linear     = "Linear",
    polynomial = sprintf("Polynomial (degree = %d, coef0 = %g)", ki$degree, ki$coef0),
    ki$type
  )
}
