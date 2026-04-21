#' Insensitivity margin in percentage units
#'
#' A dials parameter for the epsilon tube half-width in psvr MAPE models.
#' Unlike `dials::svm_margin()` which uses absolute units, this parameter
#' is expressed as a percentage of each target value. The default range
#' \[1, 20\] means the insensitivity tube spans 1% to 20% of each target.
#'
#' @param range Numeric vector of length 2. Default `c(1, 20)`.
#' @param trans A `scales` transformation object. Default `NULL`.
#'
#' @return A `quant_param` dials object.
#'
#' @examples
#' margin_percentage()
#' margin_percentage(range = c(0.5, 10))
#'
#' @export
margin_percentage <- function(range = c(1, 20), trans = NULL) {
  dials::new_quant_param(
    type      = "double",
    range     = range,
    inclusive = c(TRUE, TRUE),
    trans     = trans,
    label     = c(svm_margin = "Insensitivity Margin (%)"),
    finalize  = NULL
  )
}

#' Median-distance heuristic for RBF kernel bandwidth
#'
#' Returns the median pairwise Euclidean distance between rows of `X`,
#' which is a standard data-driven starting point for the RBF kernel
#' bandwidth (Schölkopf & Smola, 2002). Use the result to define a
#' sensible `rbf_sigma` search range centred on this value via
#' `dials::rbf_sigma(range = c(log10(sigma / 10), log10(sigma * 10)))`.
#'
#' @param X A numeric matrix or data frame of predictors (already
#'   preprocessed — centred, scaled, etc.).
#' @param sample_size Integer. If `nrow(X) > sample_size`, a random
#'   subsample is used to avoid O(n²) memory cost on large datasets.
#'   Default `500L`.
#' @param seed Integer seed for the subsample. Default `NULL`.
#'
#' @return A scalar numeric: the median pairwise Euclidean distance.
#'
#' @examples
#' X <- matrix(rnorm(200), ncol = 4)
#' sigma_heuristic(X)
#'
#' @importFrom stats dist median
#' @export
sigma_heuristic <- function(X, sample_size = 500L, seed = NULL) {
  X <- as.matrix(X)
  if (nrow(X) > sample_size) {
    if (!is.null(seed)) set.seed(seed)
    X <- X[sample(nrow(X), sample_size), , drop = FALSE]
  }
  median(dist(X))
}

#' RBF sigma parameter for psvr models
#'
#' A dials parameter for the RBF kernel bandwidth in psvr models.
#' The default range `[-3, 1]` on the log10 scale is a conservative
#' fallback. For best results, override the range using
#' [sigma_heuristic()] computed on the preprocessed training data:
#'
#' ```r
#' train_baked <- rec |> prep() |> bake(new_data = train)
#' sigma_med   <- sigma_heuristic(train_baked |> select(-outcome))
#' rbf_sigma_custom <- rbf_sigma_psvr(
#'   range = c(log10(sigma_med / 10), log10(sigma_med * 10))
#' )
#' # Then inject via option_add():
#' wf_set |> option_add(
#'   param_info = extract_parameter_set_dials(wf) |>
#'     update(rbf_sigma = rbf_sigma_custom),
#'   id = "your_workflow_id"
#' )
#' ```
#'
#' @param range Numeric vector of length 2 on the log10 scale.
#'   Default `c(-3, 1)`.
#' @param trans A `scales` transformation object.
#'   Default `scales::log10_trans()`.
#'
#' @return A `quant_param` dials object.
#'
#' @seealso [sigma_heuristic()]
#'
#' @examples
#' rbf_sigma_psvr()
#'
#' # Override with data-driven range:
#' X <- matrix(rnorm(200), ncol = 4)
#' sigma_med <- sigma_heuristic(X)
#' rbf_sigma_psvr(range = c(log10(sigma_med / 10), log10(sigma_med * 10)))
#'
#' @export
rbf_sigma_psvr <- function(range = c(-3, 1),
                            trans = scales::log10_trans()) {
  dials::new_quant_param(
    type      = "double",
    range     = range,
    inclusive = c(TRUE, TRUE),
    trans     = trans,
    label     = c(rbf_sigma = "RBF Sigma (psvr)"),
    finalize  = NULL
  )
}

#' Cost parameter with extended range for psvr models
#'
#' A dials parameter for the regularisation parameter in psvr models.
#' The default range \[-2, 10\] on the log2 scale (corresponding to
#' approximately 0.25 to 1024) is wider than `dials::cost()` to
#' accommodate the larger regularisation values typically needed by
#' LS-SVR models.
#'
#' @param range Numeric vector of length 2 on the log2 scale.
#'   Default `c(-2, 10)`.
#' @param trans A `scales` transformation object.
#'   Default `scales::log2_trans()`.
#'
#' @return A `quant_param` dials object.
#'
#' @examples
#' cost_psvr()
#'
#' @export
cost_psvr <- function(range = c(-2, 10),
                       trans = scales::log2_trans()) {
  dials::new_quant_param(
    type      = "double",
    range     = range,
    inclusive = c(TRUE, TRUE),
    trans     = trans,
    label     = c(cost = "Cost"),
    finalize  = NULL
  )
}
