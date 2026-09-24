#!/usr/bin/env python3
"""Self-test for lint_docs.py's declaration vocabulary rules.

Feeds synthetic Nim sources through `lint_text` and asserts:

- the `decl-of-suffix` rule fires on a proc, func, or template definition
  whose name ends in `Of`, call sites and prose mentions stay quiet
- the banned-vocabulary entries fire on the standalone word `fam` plus
  every prose phrase the BANNED table lists, one match per entry per line
- a line naming the GmmFamily value surface keeps the `fam` exemption

Run by hand from the tools directory:
    $ python3 test_lint_docs_decl_vocab.py
Exit status 0 on pass, 1 on fail.
"""
import sys
from pathlib import Path

from lint_docs import lint_text


def rules_at(src, name="probe.nim"):
    """Returns the (line, rule) pairs lint_text reports for one source."""
    return [(f.line, f.rule) for f in lint_text(src, name)]


def expect(label, got, want):
    """Compares one finding set, printing PASS or FAIL."""
    ok = got == want
    print(("PASS" if ok else "FAIL"), label,
          "" if ok else "got %s want %s" % (got, want))
    return ok


def main():
    """Runs the cases, printing nothing, exiting 0 on pass."""
    ok = True

    ok &= expect("proc Of-suffix fires",
                 [r for r in rules_at("proc laneCellOf(cell: int): int =\n"
                                      "  result = cell\n")
                  if r[1] == "decl-of-suffix"],
                 [(1, "decl-of-suffix")])
    ok &= expect("exported template Of-suffix fires",
                 [r for r in rules_at("template keyHeadOf*[T](x: T): int = x\n")
                  if r[1] == "decl-of-suffix"],
                 [(1, "decl-of-suffix")])
    ok &= expect("call site quiet",
                 [r for r in rules_at("proc user() =\n"
                                      "  let c = laneCellOf(3)\n"
                                      "  discard c\n")
                  if r[1] == "decl-of-suffix"], [])
    ok &= expect("prose naming the identifier quiet",
                 [r for r in rules_at(
                     "## the walk joins keyHeadOf, the shared derivation\n")
                  if r[1] == "decl-of-suffix"], [])
    ok &= expect("plain name quiet",
                 [r for r in rules_at("proc widenDtype(x: int): int = x\n")
                  if r[1] == "decl-of-suffix"], [])

    vocab = ("## the fam walks the band model over the composition tier\n"
             "## with a plane row and a frag ordering at band width 64\n"
             "## also the band-model table\n")
    got = [r for r in rules_at(vocab) if r[1] == "banned-vocab"]
    # one match per banned entry per line, each entry reports at most once per line
    ok &= expect("prose bans fire", got,
                 [(1, "banned-vocab"), (1, "banned-vocab"),
                  (2, "banned-vocab"), (3, "banned-vocab")])
    ok &= expect("fam exempt beside the GmmFamily surface",
                 [r for r in rules_at(
                     "## the GmmFamily value surface: the fam name is its own\n")
                  if r[1] == "banned-vocab"], [])
    ok &= expect("fam exempt beside gmmBf16 and gmmF16",
                 [r for r in rules_at(
                     "## gmmBf16 and gmmF16 carry the fam dtype split\n")
                  if r[1] == "banned-vocab"], [])
    ok &= expect("fam fires without the surface",
                 [r for r in rules_at("## the fam of tiles\n")
                  if r[1] == "banned-vocab"], [(1, "banned-vocab")])

    ok &= expect("gmm names quiet outright",
                 [r for r in rules_at(
                     "## GmmFamily, gmmBf16, gmmF16, gmmWiden keep their names\n")
                  if r[1] == "banned-vocab"], [])

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
