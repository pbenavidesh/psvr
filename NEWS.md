# psvr 0.1.0

Initial CRAN release.

`psvr` fits support vector regression models that optimise percentage-error
losses directly, for problems where targets are strictly positive and relative
accuracy matters more than absolute accuracy. Classical SVR minimises
absolute-error losses, which weight a fixed error equally at every scale. The
derivations are in Benavides-Herrera et al. (2026)
<doi:10.3390/math14101679>.

* Two fitters: `psvr_mape()`, an epsilon-SVR with MAPE loss, and
  `psvr_rmspe()`, a least-squares SVR with RMSPE loss. On either,
  `sym_type = "even"` or `"odd"` selects the symmetric-kernel variant, giving
  four models in total. All require `y > 0`, which is validated at fit time.
* MAPE fits are solved by a built-in SMO algorithm implemented in C++, with the
  'osqp' quadratic-programming backend available through `solver = "osqp"`.
  RMSPE fits solve an augmented linear system in base R.
* `make_kernel()` builds RBF, linear and polynomial kernels, evaluated by
  compiled code; `sigma_heuristic()` supplies a median-distance bandwidth.
  User-written kernel closures are also accepted.
* `predict()`, `fitted()`, `residuals()`, `coef()`, `summary()` and `print()`
  methods for all four fit classes.
* tidymodels integration: six parsnip specifications — `psvr_mape_rbf()`,
  `psvr_mape_poly()`, `psvr_mape_linear()` and the three `psvr_rmspe_*()`
  counterparts — registered under the `"psvr"` engine, with `sym_type` as a
  tunable argument rather than a separate specification.
* dials parameters calibrated for percentage-error models, whose useful ranges
  differ from the absolute-error defaults: `cost_psvr()`,
  `cost_psvr_ls_data()`, `margin_percentage()`, `rbf_sigma_psvr()`,
  `rbf_sigma_psvr_data()` and `sym_type_param()`; plus `psvr_option_add()` and
  `psvr_option_add_cost_ls()` to apply data-driven ranges across a workflow set.
* `psvr_cv()` runs fold-wise MAPE fits, reusing the kernel matrix across folds
  and warm-starting each fold from the previous one.

Known limitation: with the built-in SMO solver, MAPE fits on linear and
polynomial kernels may reach `max_iter` without converging and warn; the RBF
kernel is unaffected. Use `solver = "osqp"` for those combinations.

## Relationship to the archived v0.0.2

This is the first CRAN release, but not the first published version. Version
0.0.2 is archived on Zenodo (<doi:10.5281/zenodo.19935781>) and is the version
the accompanying *Mathematics* paper was computed against. The user-facing API
has changed since then. Readers reproducing the paper should install the
archived v0.0.2 rather than this release.
