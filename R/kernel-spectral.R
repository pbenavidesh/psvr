#' Adaptive spectral regularization (internal, F3)
#'
#' Implements Theorem 2 of arXiv:2605.01446 v3 with corrected eigenvalue
#' estimator. Adds a `mu * I` shift to `Omega_s` when its smallest
#' eigenvalue is numerically negative, so the SMO Hessian is provably PSD
#' (`>= delta_stab * I`). Returns the matrix untouched when already PSD.
#'
#' @section Algorithm and paper deviation:
#' Algorithm 2 line 6 of the paper, as literally written
#' (`v <- -Omega_s * v / ||Omega_s * v||`), estimates `-lambda_max(Omega_s)`
#' rather than `lambda_min(Omega_s)`: power iteration on `-Omega_s`
#' converges to the eigenvector of largest |eigenvalue|, which is
#' `v_max(Omega_s)` whenever `|lambda_max| > |lambda_min|` (the typical
#' case for Mercer-PSD or near-PSD matrices). This implementation uses the
#' standard two-pass shifted power iteration: Pass 1 estimates
#' `lambda_max` via power iteration on `Omega_s`; Pass 2 estimates
#' `lambda_min` via power iteration on `rho * I - Omega_s` (whose
#' dominant eigenvector is `v_min(Omega_s)`). Both passes are O(N^2); the
#' total cost is `2 * T_pi` matvecs.
#'
#' The Pass 2 shift uses the spectral radius `rho = |Pass 1 Rayleigh|`,
#' not Pass 1's signed Rayleigh. This handles the `lambda_min`-dominant
#' pathological case (`|lambda_min| > lambda_max`), where plain power
#' iteration on `Omega_s` in Pass 1 converges to `v_min` and gives a
#' negative Rayleigh; using `abs(Pass 1 Rayleigh)` ensures the shifted
#' operator `rho * I - Omega_s` is PSD, so Pass 2 reliably finds
#' `v_min(Omega_s)` regardless of which side of the spectrum dominated
#' Pass 1.
#'
#' @section Determinism:
#' Both passes start from the uniform unit vector `rep(1, N) / sqrt(N)`
#' (not random) so the routine is bit-reproducible.
#'
#' @param Omega_s Symmetrized kernel matrix
#'   `Omega_s = 1/2 (Omega + a * Omega^*)`, already with any caller-applied
#'   jitter.
#' @param T_pi Power-iteration steps per pass (default `5L`).
#' @param delta_stab Numerical PSD floor (default `1e-8`).
#' @return A list with components:
#'   \describe{
#'     \item{`Omega_use`}{Matrix to pass to the SMO/QP solver.}
#'     \item{`mu`}{Numeric scalar; the shift applied (`0` if no shift).}
#'     \item{`lambda_min_hat`}{Numeric scalar; Rayleigh-quotient estimate
#'       of `lambda_min(Omega_s)` from Pass 2.}
#'     \item{`lambda_max_hat`}{Numeric scalar; Pass 1's Rayleigh quotient.
#'       For PSD or `lambda_max`-dominant matrices (the typical case for
#'       Mercer kernels in this package), this equals
#'       `lambda_max(Omega_s)`. For `lambda_min`-dominant matrices
#'       (`|lambda_min| > lambda_max` — a pathological case not reachable
#'       with `make_kernel()`-supplied kernels), it equals
#'       `lambda_min(Omega_s)`. The branch decision in either case is made
#'       correctly via the Pass 2 estimate of `lambda_min_hat`.}
#'     \item{`branch_taken`}{Character `"no_shift"` or `"shifted"`.}
#'     \item{`n_power_iterations`}{Integer vector of length 2; iterations
#'       executed in Pass 1 and Pass 2.}
#'   }
#' @keywords internal
.adaptive_spectral_shift <- function(Omega_s, T_pi = 5L, delta_stab = 1e-8) {
  N  <- nrow(Omega_s)
  v0 <- rep(1, N) / sqrt(N)

  # ===== Pass 1: power iteration on Omega_s to estimate lambda_max =====
  v     <- v0
  iter1 <- 0L
  for (t in seq_len(T_pi)) {
    Av      <- as.numeric(Omega_s %*% v)
    Av_norm <- sqrt(sum(Av * Av))
    if (Av_norm < .Machine$double.eps) {
      # Omega_s is the zero matrix (up to floating-point); no shift needed.
      return(list(
        Omega_use          = Omega_s,
        mu                 = 0,
        lambda_min_hat     = 0,
        lambda_max_hat     = 0,
        branch_taken       = "no_shift",
        n_power_iterations = c(t, 0L)
      ))
    }
    v     <- Av / Av_norm
    iter1 <- t
  }
  lambda_max_hat <- as.numeric(crossprod(v, Omega_s %*% v))
  # Spectral radius. abs() is essential for the lambda_min-dominant case
  # (|lambda_min| > lambda_max), where Pass 1 converges to v_min and
  # lambda_max_hat is negative; without abs(), the Pass 2 operator would
  # NOT be PSD and would converge to v_max instead of v_min.
  rho_hat <- abs(lambda_max_hat)

  # ===== Pass 2: power iteration on (rho_hat * I - Omega_s) =====
  # All eigenvalues of (rho_hat * I - Omega_s) are >= 0 because
  # rho_hat >= lambda_max(Omega_s); the dominant eigenvalue is
  # (rho_hat - lambda_min) and its eigenvector is v_min(Omega_s). The
  # Rayleigh quotient v^T Omega_s v on the converged v then yields
  # lambda_min_hat.
  v              <- v0
  iter2          <- 0L
  lambda_min_hat <- NA_real_
  for (t in seq_len(T_pi)) {
    Av      <- rho_hat * v - as.numeric(Omega_s %*% v)
    Av_norm <- sqrt(sum(Av * Av))
    if (Av_norm < .Machine$double.eps) {
      # Degenerate: lambda_max ~= lambda_min (Omega_s ~= scalar * I).
      lambda_min_hat <- lambda_max_hat
      iter2          <- t
      break
    }
    v     <- Av / Av_norm
    iter2 <- t
  }
  if (is.na(lambda_min_hat)) {
    lambda_min_hat <- as.numeric(crossprod(v, Omega_s %*% v))
  }

  # ===== Branch decision =====
  if (lambda_min_hat >= -delta_stab) {
    list(
      Omega_use          = Omega_s,
      mu                 = 0,
      lambda_min_hat     = lambda_min_hat,
      lambda_max_hat     = lambda_max_hat,
      branch_taken       = "no_shift",
      n_power_iterations = c(iter1, iter2)
    )
  } else {
    mu              <- -lambda_min_hat + delta_stab
    Omega_use       <- Omega_s
    diag(Omega_use) <- diag(Omega_use) + mu
    list(
      Omega_use          = Omega_use,
      mu                 = mu,
      lambda_min_hat     = lambda_min_hat,
      lambda_max_hat     = lambda_max_hat,
      branch_taken       = "shifted",
      n_power_iterations = c(iter1, iter2)
    )
  }
}
