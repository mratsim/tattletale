#!/usr/bin/env python3
"""Refusal-slop rule fixtures over lint_md_nim_py.py.

Test that the convicted refusal specimens fire their rules and the honest
one-liner refusals stay clean.

Run:

    python3 test_lint_refusal.py
    python3 lint_md_nim_py.py --selftest-refusal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint_md_nim_py  # noqa: E402  (the sys.path setup runs above)


def test_refusal_rules():
    """Every fixture case reports exactly its expected findings.

    Returns the failed case count through the assertion message.
    """

    failed = lint_md_nim_py.refusal_selftest(quiet=True)
    assert failed == 0, "%d refusal fixture case(s) failed" % failed


if __name__ == "__main__":
    failed = lint_md_nim_py.refusal_selftest()
    sys.exit(1 if failed else 0)
