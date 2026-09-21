#!/usr/bin/env python3
"""Self-check fixture for lint_docs.py's `#[ ... ]#` block-comment exclusion.

Feeds a synthetic Nim source through `lint_text` and asserts two facts:
- zero findings attributed to lines inside a block comment, whatever prose
  violations the interior carries (the-opener, semicolon, banned vocabulary,
  em-dash)
- at least one finding on a real doc-comment line carrying the same violations,
  so the empty interior result is the exclusion working, not the rules idling

Run by hand from the tools directory:
    $ python3 test_lint_docs_block_comment.py
Exit status 0 on pass, 1 on fail.
"""
import sys
from pathlib import Path

from lint_docs import lint_text

NIM_SRC = """\
## The module holds a window; the runner walks it -- one loop.
proc render() =
  ## The second comment; also violating -- yet counted.
  discard

#[
  The interior comment below carries violations on purpose, the exclusion
  under test dropping them:
  ## The window holds the text; the interior is commented out -- one loop.
  ## The pins hold the line, because the tests demand it -- a design story.
]#
"""


def main():
    """Runs the two assertions, printing nothing, exiting 0 on pass."""
    findings = lint_text(NIM_SRC, "fixture_block_comment.nim")
    interior = [f for f in findings if 9 <= f.line <= 11]
    assert not interior, "block-comment interior counted findings: %r" % interior
    counted = [f for f in findings if f.rule not in
               ("wall-of-text", "wall-no-air", "missing-diagram", "how-narration")]
    assert counted, "no finding on the violating doc lines, the exclusion idles the rules"
    return 0


if __name__ == "__main__":
    sys.exit(main())
