# Apply data-driven rbf_sigma to all psvr workflows in a workflow set

A convenience wrapper that calls
[`workflowsets::option_add()`](https://workflowsets.tidymodels.org/reference/option_add.html)
for every psvr workflow in `wf_set` (those whose `wflow_id` contains
`"m1"`, `"m2"`, `"m3"`, or `"m4"`), replacing the `rbf_sigma` dials
parameter with a data-driven one built from `X` via
[`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md).

## Usage

``` r
psvr_option_add(wf_set, X, width = 10, sample_size = 500L, seed = NULL)
```

## Arguments

- wf_set:

  A `workflow_set` object.

- X:

  A numeric matrix or data frame of preprocessed predictors.

- width:

  Positive scalar. Passed to
  [`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md).
  Default `10`.

- sample_size:

  Integer. Passed to
  [`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md).
  Default `500L`.

- seed:

  Integer seed for subsampling. Default `NULL`.

## Value

The updated `workflow_set` (the same object with `option_add()` applied
to each psvr workflow).

## See also

[`rbf_sigma_psvr_data()`](https://pbenavidesh.github.io/psvr/reference/rbf_sigma_psvr_data.md),
[`sigma_heuristic()`](https://pbenavidesh.github.io/psvr/reference/sigma_heuristic.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# After building wf_set and preprocessing:
train_baked <- rec |> prep() |> bake(new_data = train)
wf_set <- psvr_option_add(wf_set, train_baked |> select(-outcome))
} # }
```
