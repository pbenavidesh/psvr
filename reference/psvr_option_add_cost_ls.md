# Apply data-driven LS-SVR cost range to all m3/m4 workflows in a workflow set

A convenience wrapper that calls
[`workflowsets::option_add()`](https://workflowsets.tidymodels.org/reference/option_add.html)
for every LS-SVR psvr workflow in `wf_set` (those whose `wflow_id`
matches `"m3"` or `"m4"`), replacing the `cost` dials parameter with one
built from `y` via
[`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md).

## Usage

``` r
psvr_option_add_cost_ls(wf_set, y, width_log2 = 4)
```

## Arguments

- wf_set:

  A `workflow_set` object.

- y:

  Numeric vector of strictly positive training targets.

- width_log2:

  Passed to
  [`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md).
  Default `4`.

## Value

The updated `workflow_set`.

## Details

Workflows for `m1`/`m2` (`ε`-SVR) are intentionally skipped — for those,
`cost` maps to `C` and the static
[`cost_psvr()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr.md)
range is usually adequate.

Note:
[`workflowsets::option_add()`](https://workflowsets.tidymodels.org/reference/option_add.html)
replaces the whole `param_info` option for each matched workflow. If you
also need a data-driven `rbf_sigma` (via
[`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)),
build the full `param_info` manually with
[`tune::extract_parameter_set_dials()`](https://tune.tidymodels.org/reference/reexports.html)
and call
[`workflowsets::option_add()`](https://workflowsets.tidymodels.org/reference/option_add.html)
in one shot, or call this helper first and then
[`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)
(which only touches `rbf_sigma`) — the latter currently overwrites the
former, so prefer the manual one-shot approach when both are needed.

## See also

[`cost_psvr_ls_data()`](https://pbenavidesh.github.io/psvr/reference/cost_psvr_ls_data.md),
[`psvr_option_add()`](https://pbenavidesh.github.io/psvr/reference/psvr_option_add.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# After building wf_set with at least one m3 or m4 workflow:
wf_set <- psvr_option_add_cost_ls(wf_set, y = train_df$y)
} # }
```
