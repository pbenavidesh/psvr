## Test environments

* win-builder, R-devel (2026-08-10 r90389 ucrt), Windows Server 2022 —
  0 errors | 0 warnings | 1 note.
* GitHub Actions, `R CMD check --as-cran`: macOS-latest (release),
  Windows-latest (release), Ubuntu-latest (devel, release, oldrel-1) —
  all five OK. These jobs run with `--no-manual`, so the PDF manual is
  validated on win-builder rather than there.
* R-hub (Linux, macOS) — PENDING; results not yet available.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

* Possibly misspelled words in DESCRIPTION:

      Benavides (14:46)
      LS (10:60, 13:66)
      MAPE (10:22)
      RMSPE (10:72)
      SMO (12:54)
      SVR (10:13, 10:63, 12:17, 13:69)
      al (14:67)
      et (14:64)

  All eight are spelled as intended. Five are standard terminology in the
  statistics and machine-learning literature:

  * SVR — support vector regression
  * MAPE — mean absolute percentage error
  * RMSPE — root mean squared percentage error
  * LS — least squares (in "LS-SVR", least-squares support vector regression)
  * SMO — sequential minimal optimization

  The other three come from the citation "Benavides-Herrera et al. (2026)":
  "et" and "al" are the usual Latin abbreviation, and "Benavides" is the
  first half of the author's hyphenated surname, which the spell checker
  splits at the hyphen ("Herrera", at 14:56, is not flagged).

## Downstream dependencies

There are currently no downstream dependencies for this package.
