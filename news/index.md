# Changelog

## psvr 0.1.0

Initial CRAN release.

`psvr` fits support vector regression models that optimise
percentage-error losses directly, for problems where targets are
strictly positive and relative accuracy matters more than absolute
accuracy. Classical SVR minimises absolute-error losses, which weight a
fixed error equally at every scale. The derivations are in
Benavides-Herrera et al. (2026) <doi:10.3390/math14101679>.

- Two fitters:
  [`psvr_mape()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape.md),
  an epsilon-SVR with MAPE loss, and
  [`psvr_rmspe()`](https://pbenavidesh.github.io/psvr/reference/psvr_rmspe.md),
  a least-squares SVR with RMSPE loss. On either, `sym_type = "even"` or
  `"odd"` selects the symmetric-kernel variant, giving four models in
  total. All require `y > 0`, which is validated at fit time.
- MAPE fits are solved by a built-in SMO algorithm implemented in C++,
  with the ‘osqp’ quadratic-programming backend available through
  `solver = "osqp"`. RMSPE fits solve an augmented linear system in base
  R.
- [`make_kernel()`](https://pbenavidesh.github.io/psvr/reference/make_kernel.md)
  builds RBF, linear and polynomial kernels, evaluated by compiled code;
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)
  supplies a median-distance bandwidth. User-written kernel closures are
  also accepted.
- [`predict()`](https://rdrr.io/r/stats/predict.html),
  [`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
  [`residuals()`](https://rdrr.io/r/stats/residuals.html),
  [`coef()`](https://rdrr.io/r/stats/coef.html),
  [`summary()`](https://rdrr.io/r/base/summary.html) and
  [`print()`](https://rdrr.io/r/base/print.html) methods for all four
  fit classes.
- tidymodels integration: six parsnip specifications —
  [`psvr_mape_rbf()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md),
  [`psvr_mape_poly()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md),
  [`psvr_mape_linear()`](https://pbenavidesh.github.io/psvr/reference/psvr_mape_specs.md)
  and the three `psvr_rmspe_*()` counterparts — registered under the
  `"psvr"` engine, with `sym_type` as a tunable argument rather than a
  separate specification.
- dials parameters calibrated for percentage-error models, whose useful
  ranges differ from the absolute-error defaults:
  [`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md),
  [`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md),
  [`margin_percentage()`](https://pbenavidesh.github.io/psvr/reference/margin_percentage.md),
  [`rbf_sigma_psvr()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr.md),
  [`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md)
  and
  [`sym_type_param()`](https://pbenavidesh.github.io/psvr/reference/sym_type_param.md);
  plus
  [`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)
  and
  [`psvr_option_add_cost_ls()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add_cost_ls.md)
  to apply data-driven ranges across a workflow set.
- [`psvr_cv()`](https://pbenavidesh.github.io/psvr/reference/psvr_cv.md)
  runs fold-wise MAPE fits, reusing the kernel matrix across folds and
  warm-starting each fold from the previous one.

Known limitation: with the built-in SMO solver, MAPE fits on linear and
polynomial kernels may reach `max_iter` without converging and warn; the
RBF kernel is unaffected. Use `solver = "osqp"` for those combinations.

### Relationship to the archived v0.0.2

This is the first CRAN release, but not the first published version.
Version 0.0.2 is archived on Zenodo (<doi:10.5281/zenodo.19935781>) and
is the version the accompanying *Mathematics* paper was computed
against. The user-facing API has changed since then. Readers reproducing
the paper should install the archived v0.0.2 rather than this release.
