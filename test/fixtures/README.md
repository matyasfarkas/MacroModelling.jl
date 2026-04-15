# Test Fixtures

This directory is the canonical home for package-test fixtures used by `test/`.

During consolidation, the large SEP/Dynare validation corpus remains stored under the
legacy path `tests/sep_validation` and is exposed here via a local symlink:

- `test/fixtures/sep_validation -> ../../tests/sep_validation`

Scripts and tests should prefer `test/fixtures/sep_validation` and may fall back to
the legacy path while the historical tree is being fully reclassified.
