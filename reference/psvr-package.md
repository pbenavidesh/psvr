# psvr: Percentage-Error Support Vector Regression

Implements four support vector regression models derived from a unified
mathematical framework for percentage-error loss functions: epsilon-SVR
with MAPE, its symmetric kernel extension, LS-SVR with RMSPE, and its
symmetric counterpart. All models require strictly positive targets. The
epsilon-SVR models are solved via a built-in SMO algorithm (with osqp
available as an optional alternative backend) and the LS-SVR models via
a linear system (base R). See Benavides-Herrera et al. (2026, under
review at MDPI Mathematics) for the mathematical derivations.

## See also

Useful links:

- <https://pbenavidesh.github.io/psvr/>

## Author

**Maintainer**: Pablo Benavides-Herrera <pbenavides@iteso.mx>
([ORCID](https://orcid.org/0000-0003-4926-4763))
