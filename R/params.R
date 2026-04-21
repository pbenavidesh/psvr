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

#' RBF sigma parameter with data-driven range for psvr models
#'
#' A convenience wrapper that combines [sigma_heuristic()] and
#' [rbf_sigma_psvr()] in a single call. It computes the median pairwise
#' distance from `X` and returns a `quant_param` whose search range spans
#' one order of magnitude either side of the heuristic value on the log10
#' scale (i.e., `[log10(sigma_med / width), log10(sigma_med * width)]`).
#'
#' @param X A numeric matrix or data frame of predictors (already
#'   preprocessed — centred, scaled, etc.).
#' @param width Positive scalar. Multiplier that sets the half-width of the
#'   search range around the heuristic sigma. Default `10` (one decade).
#' @param sample_size Integer. Passed to [sigma_heuristic()]. Default `500L`.
#' @param seed Integer seed for subsampling. Passed to [sigma_heuristic()].
#'   Default `NULL`.
#' @param trans A `scales` transformation object. Default `scales::log10_trans()`.
#'
#' @return A `quant_param` dials object with a data-driven search range.
#'
#' @seealso [sigma_heuristic()], [rbf_sigma_psvr()]
#'
#' @examples
#' X <- matrix(rnorm(200), ncol = 4)
#' rbf_sigma_psvr_data(X)
#' rbf_sigma_psvr_data(X, width = 5)
#'
#' @export
rbf_sigma_psvr_data <- function(X, width = 10, sample_size = 500L,
                                 seed = NULL,
                                 trans = scales::log10_trans()) {
  sigma_med <- sigma_heuristic(X, sample_size = sample_size, seed = seed)
  rbf_sigma_psvr(
    range = c(log10(sigma_med / width), log10(sigma_med * width)),
    trans = trans
  )
}

#' Apply data-driven rbf_sigma to all psvr workflows in a workflow set
#'
#' A convenience wrapper that calls [workflowsets::option_add()] for every
#' psvr workflow in `wf_set` (those whose `wflow_id` contains `"m1"`,
#' `"m2"`, `"m3"`, or `"m4"`), replacing the `rbf_sigma` dials parameter
#' with a data-driven one built from `X` via [rbf_sigma_psvr_data()].
#'
#' @param wf_set A `workflow_set` object.
#' @param X A numeric matrix or data frame of preprocessed predictors.
#' @param width Positive scalar. Passed to [rbf_sigma_psvr_data()].
#'   Default `10`.
#' @param sample_size Integer. Passed to [sigma_heuristic()]. Default `500L`.
#' @param seed Integer seed for subsampling. Default `NULL`.
#'
#' @return The updated `workflow_set` (the same object with `option_add()`
#'   applied to each psvr workflow).
#'
#' @seealso [rbf_sigma_psvr_data()], [sigma_heuristic()]
#'
#' @examples
#' \dontrun{
#' # After building wf_set and preprocessing:
#' train_baked <- rec |> prep() |> bake(new_data = train)
#' wf_set <- psvr_option_add(wf_set, train_baked |> select(-outcome))
#' }
#'
#' @importFrom stats update
#' @export
psvr_option_add <- function(wf_set, X, width = 10, sample_size = 500L,
                             seed = NULL) {
  rbf_param <- rbf_sigma_psvr_data(X, width = width,
                                    sample_size = sample_size, seed = seed)
  psvr_ids <- wf_set$wflow_id[grepl("m1|m2|m3|m4", wf_set$wflow_id)]
  for (id in psvr_ids) {
    param_info <- workflowsets::extract_workflow(wf_set, id = id) |>
      tune::extract_parameter_set_dials() |>
      update(rbf_sigma = rbf_param)
    wf_set <- workflowsets::option_add(wf_set, param_info = param_info, id = id)
  }
  wf_set
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
