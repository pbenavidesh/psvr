# Test-suite-wide options. Run by testthat before any test file, and by
# `R CMD check`'s test phase. NOT run by devtools::load_all().

# Surface `$` partial matching on lists as a warning.
#
# WHY. The four fit classes carry different field vectors (16 / 18 / 10 / 11
# names), so a `$` read that is correct on one can silently return a DIFFERENT
# field on another. `$` does exact match first, then unique partial match. The
# case that motivated this: `summary()` tested for symmetry with
# `!is.null(object$a)`; on `psvr_rmspe` there is no `a`, `a` uniquely prefixes
# `alpha`, and the caller got the length-N multiplier vector instead of NULL.
# It was correct on two of the four classes for a reason unrelated to the field
# being absent. Nothing catches this class of defect: the value is wrong, not
# missing, so no error is raised, `R CMD check` is green, and the suite passes.
#
# The static side of this is closed by derivation (PSVR_STATUS.md 9.21-A): over
# the 21 field names the wrong-object surface is exactly one cell,
# `psvr_rmspe$a -> alpha`. This option is the dynamic guard behind that
# derivation -- it covers what the derivation cannot, namely abbreviations
# (`$conv`, `$fitt`: 274 such strings resolve today, all to the right field, so
# they are a fragility hazard) and any future change to the four shapes.
#
# Measured zero warnings across the suite when added; see the commit body.
#
# CAVEAT for anyone consuming the output -- read PSVR_STATUS.md 5, "Match the
# condition object, never the message text". R emits this warning in the session
# locale. On the maintainer's machine that is Spanish, and the translation is
# PARTIAL: the header is `Aviso:`, not `Warning message:`, and the body reads
# `encuentros parciales de 'a' to 'alpha'` -- leading phrase translated, the
# connective `to` left in English. A watcher grepping the English
# "partial match of" matches nothing and reports clean. Match on the condition
# object, or pin LANGUAGE=en.
#
# The four protected snapshot files are NOT exposed to warnings emitted here:
# all four use expect_snapshot_value(style = "serialize"), which captures a
# serialized value rather than console output. Verified by reading and by MD5.
options(warnPartialMatchDollar = TRUE)
