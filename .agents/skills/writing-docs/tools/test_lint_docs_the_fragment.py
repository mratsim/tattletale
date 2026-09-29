#!/usr/bin/env python3
"""Self-check fixture for lint_docs.py's the-fragment rule.

Feeds synthetic prose through `lint_text` and asserts three invariants:
- a comma-tail and a mid-line sentence-initial "the" noun phrase without
  any verb each produce exactly one the-fragment finding on their line
- phrases with a verb (auxiliary or third-person -s) produce zero the-fragment findings
- a noun phrase opening on "a" produces zero the-fragment findings, since the rule covers the/the only

Run by hand from the tools directory:
    $ python3 test_lint_docs_the_fragment.py
Exit status 0 on pass, 1 on fail.
"""
import sys
from pathlib import Path

from lint_docs import lint_text

NIM_SRC = """\
proc render() =
  ## Wrapping starts here.
  ## The quotient runs unmod'd at the largest stride, the documented divergence.
  ## The layout stays compact, and the excess accumulates at the tail.
  ## Wrapping starts here. The documented divergence.
  ## Wrapping starts here. The quotient is unmod'd at the tail.
  ## Wrapping starts at the largest stride, a rule with no exception.
  discard
"""


def main():
    """Runs the three assertions, printing nothing, exiting 0 on pass."""
    findings = lint_text(NIM_SRC, "fixture_the_fragment.nim")
    frag = {f.line for f in findings if f.rule == "the-fragment"}
    assert frag == {3, 5}, "the-fragment fired on %r" % sorted(frag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
