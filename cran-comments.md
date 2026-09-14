## Resubmission

This is a resubmission of a first submission. The two points raised in
review are addressed as follows.

* Acronyms. All five acronyms are now expanded at first use in the
  Description: support vector regression (SVR), mean absolute percentage
  error (MAPE), least-squares SVR (LS-SVR), root mean square percentage
  error (RMSPE), and sequential minimal optimization (SMO).

* Examples. The `\dontrun` blocks on the two parsnip specification pages
  (`psvr_mape_specs`, `psvr_rmspe_specs`) have been unwrapped, not
  converted to `\donttest`: they only construct model specifications and
  run in about 0.02 seconds each, as the check's example output confirms.

  Two `\dontrun` blocks remain, on `psvr_option_add` and
  `psvr_option_add_cost_ls`. Both examples require a pre-existing
  `workflow_set` object, and constructing one depends on the workflows and
  recipes packages, which are in Suggests rather than Imports, so neither
  example is self-contained. `\donttest` is not a usable substitute
  because it executes during `R CMD check`, while these examples reference
  objects a reader would have built beforehand. They fall under the
  documented exception for examples that genuinely cannot be executed.

## Test environments

* win-builder, R Under development (unstable) (2026-09-13 r90534 ucrt),
  Windows Server 2022 x64 (build 20348), x86_64-w64-mingw32, on the
  submitted tree (65acd6a) — 0 errors | 0 warnings | 1 note. Tests, the
  PDF and HTML manuals, vignettes, and examples all OK.
* GitHub Actions, `R CMD check --as-cran`: macOS-latest (release),
  Windows-latest (release), Ubuntu-latest (devel, release, oldrel-1) —
  all five OK. These jobs run with `--no-manual`; the PDF manual is
  validated on win-builder and on R-hub.
* R-hub v2, `R CMD check --as-cran` — three platforms, each on its own
  R-devel revision, all three `Status: OK`: x86_64-pc-linux-gnu,
  R-devel r90185 (2026-06-21); x86_64-w64-mingw32, R-devel r90447
  (2026-08-25); x86_64-apple-darwin20, R-devel r90449 (2026-08-27).
  These three runs were on the previous tree (3a0c917), not the current
  one. They were not repeated because the only changes since are the
  Description text and the example wrappers in two Rd files; no code
  changed.

## R CMD check results

0 errors | 0 warnings | 1 note

The note is from win-builder on the current tree. R-hub returned
`Status: OK` on all three platforms, and the five GitHub Actions jobs are
clean.

* This is a new submission.

* Possibly misspelled words in DESCRIPTION:

      Benavides (17:9)
      LS (11:64, 16:34)
      MAPE (11:6)
      RMSPE (12:55)
      SMO (15:19)
      SVR (8:57, 10:24, 11:59, 11:67, 14:17, 16:37)
      al (17:30)
      et (17:27)

  Every flagged word is spelled as intended (the exact set varies slightly
  by platform, as the spell-check dictionaries differ). Five are standard
  terminology in the statistics and machine-learning literature, and each
  is now expanded at first use in the Description:

  * SVR — support vector regression
  * MAPE — mean absolute percentage error
  * RMSPE — root mean square percentage error
  * LS — least squares (in "LS-SVR", least-squares support vector regression)
  * SMO — sequential minimal optimization

  The other three come from the citation "Benavides-Herrera et al. (2026)":
  "et" and "al" are the usual Latin abbreviation, and "Benavides" is the
  first half of the author's hyphenated surname, which the spell checker
  splits at the hyphen ("Herrera", at 17:19, is not flagged).

## Downstream dependencies

There are currently no downstream dependencies for this package.
