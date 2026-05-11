// psvr — Rcpp binding for psvr::smo_fit.
//
// The R-side .smo_solve() dispatcher (engine = "rcpp") calls
// psvr_smo_fit_rcpp() with:
//   - Omega       : NumericMatrix (column-major, already jittered /
//                   spectrally shifted upstream)
//   - y           : NumericVector
//   - opts        : named List containing all FitOptions fields. The
//                   parse_fit_options() helper extracts each named
//                   entry with a sensible default if missing.
//
// Returns a List matching the R-level .smo_solve_r() return shape so
// the two engines are interchangeable downstream.

#include <Rcpp.h>
#include "core_smo_solve.h"

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::IntegerVector;
using Rcpp::LogicalVector;
using Rcpp::Named;
using Rcpp::wrap;
using Rcpp::as;

namespace {

inline bool has(const List& opts, const char* name) {
  return opts.containsElementNamed(name) && !Rf_isNull(opts[name]);
}

inline double get_scalar_double(const List& opts, const char* name, double dflt) {
  if (!has(opts, name)) return dflt;
  return as<double>(opts[name]);
}

inline psvr::Index get_scalar_index(const List& opts, const char* name, psvr::Index dflt) {
  if (!has(opts, name)) return dflt;
  return static_cast<psvr::Index>(as<int>(opts[name]));
}

inline bool get_scalar_bool(const List& opts, const char* name, bool dflt) {
  if (!has(opts, name)) return dflt;
  return as<bool>(opts[name]);
}

inline psvr::Vec get_vec_double(const List& opts, const char* name) {
  if (!has(opts, name)) return psvr::Vec{};
  NumericVector v = opts[name];
  return psvr::Vec(v.begin(), v.end());
}

inline psvr::BoolVec get_vec_bool(const List& opts, const char* name) {
  if (!has(opts, name)) return psvr::BoolVec{};
  LogicalVector v = opts[name];
  psvr::BoolVec out(v.size());
  for (R_xlen_t k = 0; k < v.size(); ++k) {
    out[k] = (v[k] == TRUE) ? 1u : 0u;
  }
  return out;
}

psvr::FitOptions parse_fit_options(const List& opts) {
  psvr::FitOptions fo;
  fo.C                 = get_scalar_double(opts, "C",                 1.0);
  fo.eps               = get_scalar_double(opts, "eps",               0.1);
  fo.tol               = get_scalar_double(opts, "tol",               1e-3);
  fo.max_iter          = get_scalar_index (opts, "max_iter",          100000);
  fo.n_check           = get_scalar_index (opts, "n_check",           -1);
  fo.n_freeze          = get_scalar_index (opts, "n_freeze",          5);
  fo.block_k4_enabled  = get_scalar_bool  (opts, "block_k4_enabled",  true);
  fo.alpha_couple      = get_scalar_double(opts, "alpha_couple",      0.5);
  fo.warm_start_check  = get_scalar_bool  (opts, "warm_start_check",  false);
  fo.alpha_init        = get_vec_double   (opts, "alpha_init");
  fo.alpha_star_init   = get_vec_double   (opts, "alpha_star_init");
  fo.new_mask          = get_vec_bool     (opts, "new_mask");
  fo.trace             = get_scalar_bool  (opts, "trace",             false);
  return fo;
}

}  // anonymous namespace

// [[Rcpp::export]]
List psvr_smo_fit_rcpp(const NumericMatrix& Omega,
                        const NumericVector& y,
                        const List& opts) {
  const psvr::Index N = static_cast<psvr::Index>(Omega.nrow());
  if (Omega.ncol() != N) {
    Rcpp::stop("Omega must be square (N x N)");
  }
  if (static_cast<psvr::Index>(y.size()) != N) {
    Rcpp::stop("length(y) must equal nrow(Omega)");
  }

  const psvr::FitOptions fo = parse_fit_options(opts);
  const psvr::FitResult  r  = psvr::smo_fit(REAL(Omega), N, REAL(y), fo);

  // Convert -1.0 sentinel → NA_real_ for the decoupling-rate fields.
  const double dr   = (r.decoupling_rate              < 0) ? NA_REAL : r.decoupling_rate;
  const double dr_e = (r.early_phase_decoupling_rate  < 0) ? NA_REAL : r.early_phase_decoupling_rate;
  const double dr_l = (r.late_phase_decoupling_rate   < 0) ? NA_REAL : r.late_phase_decoupling_rate;

  // F7.5 — delta_history: NULL when trace=FALSE, numeric(iter) when trace=TRUE
  // (possibly numeric(0) if max_iter=0). Matches .smo_solve_r() shape.
  SEXP dh = fo.trace
              ? Rcpp::wrap(NumericVector(r.delta_history.begin(),
                                         r.delta_history.end()))
              : R_NilValue;

  return List::create(
    Named("alpha")                       = NumericVector(r.alpha.begin(),      r.alpha.end()),
    Named("alpha_star")                  = NumericVector(r.alpha_star.begin(), r.alpha_star.end()),
    Named("b")                           = r.b,
    Named("converged")                   = r.converged,
    Named("iterations")                  = static_cast<int>(r.iterations),
    Named("joint_updates")               = static_cast<int>(r.joint_updates),
    Named("k2_fallbacks")                = static_cast<int>(r.k2_fallbacks),
    Named("decoupling_rate")             = dr,
    Named("early_phase_decoupling_rate") = dr_e,
    Named("late_phase_decoupling_rate")  = dr_l,
    Named("delta_history")               = dh
  );
}
