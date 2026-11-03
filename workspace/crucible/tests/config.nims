# Pass-boundary verification runs with every test-tree compilation.
# The corpus and suites gate the verifier: clean input must verify clean,
# and a gating bug surfaces as spurious refusals in any suite, not silently.
--define:CIR_verify
