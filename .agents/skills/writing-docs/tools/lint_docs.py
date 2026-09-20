#!/usr/bin/env python3
"""Doc-comment linter for the writing-docs skill.
- deterministic checks over Nim doc comments (## and /// blocks)
- maintainer # comments, Python docstrings, Python # comments
- Markdown prose

L2 to L4 import this module (lint_text, RULES) as their single
source of truth for the doc rules.

Rule table (rule | trigger | severity):

| rule-id              | trigger                                                                                           | severity |
| -------------------- | ------------------------------------------------------------------------------------------------- | -------- |
| banned-vocab         | a blocklist word (EXAMPLES.md, plus operator-extended entries)                                    | counted  |
| the-opener           | a doc comment, maintainer comment or heading opens with the article "The"                         | counted  |
| semicolon            | a semicolon in prose                                                                              | counted  |
| em-dash              | an em-dash or en-dash in prose                                                                    | counted  |
| line-length          | a prose line over 140 characters                                                                  | counted  |
| article-eol          | a line ends on a dangling article or a stranded possessive                                        | counted  |
| stray-fragment       | a line ends on a bare connective or a fragment after the period                                   | counted  |
| colon-break          | a colon orphaned at line start or split from its lead phrase                                      | counted  |
| colon-inline         | a prose colon followed by prose on the same line                                                  | counted  |
| unit-split           | a line opens on a severed one-word continuation ("apply,")                                        | counted  |
| paren-split          | a line ends inside an open parenthesis                                                            | counted  |
| single-word-eol      | a 1-2 word stub line with reflow room on the previous line                                        | counted  |
| doc-above-type       | a ## block sits directly above a type declaration                                                 | counted  |
| bullet-list-length   | a bullet list with 4 or more items                                                                | counted  |
| bullet-item-length   | a single bullet item spanning 4 or more lines                                                     | counted  |
| table-separator      | a table with no |---| separator row after the header                                              | counted  |
| table-mispadding     | a table row with a different cell count than the header, or a doc table row with an unpadded cell | counted  |
| table-cell-wall      | a table cell over 30 words                                                                        | counted  |
| table-alignment      | a table whose pipe separators do not line up (pad each cell to the column width)                  | counted  |
| escape-noise         | an escape sequence used as a prose word                                                           | counted  |
| narration            | temporal or history narration: currently, as of, once X lands                                     | counted  |
| artifact-ref         | a pipeline artifact ID: SLOP-002, QA-004, iter-3                                                  | counted  |
| wall-of-text         | 10+ consecutive prose lines with no bullet, table, or diagram                                     | advisory |
| wall-no-air          | the longest consecutive airless prose stretch reaches 4 lines                                     | counted  |
| missing-diagram      | a multi-step flow described in prose or bullets with no diagram                                   | advisory |
| how-narration        | a doc narrates the how: "This function...", "we iterate", "make sure"                             | advisory |
| module-header-length | a module header past 8 tight prose lines                                                          | advisory |
| test-header-command  | a test file header with no run command line                                                       | counted  |
| missing-doc          | a public item with no doc comment (exported Nim proc or type, module-level Python def or class)   | counted  |
| missing-contract     | a multi-line function doc with no contract marker (Args, Returns, Contract, Invariant)            | advisory |
| sig-wrap             | a proc or func signature wrapped across lines while the joined form fits a 140-char line          | counted  |
| except-rewrap        | an except clause re-raises the caught exception (rewrap)                                          | counted  |
| try-block            | try/except or try/finally catching as control flow outside the libtorch C++ boundary and tests    | counted  |
| design-narration     | a doc or maintainer comment justifying the design choice instead of stating the contract (because, X and not Y) | counted  |
| section-separator    | a whole-line `#` comment built from dashes (a layout-position marker)                             | counted  |

Golden rules:
- ## docs serve API users, # comments serve maintainers and auditors
- write what the code does, its preconditions, its invariants, never the journey
- every sentence stands readable to a fresh clone of the repository
- prefer bullets, tables, and diagrams over dense prose runs
- both skill files are the standard, read them first, they are always forgotten:
  - repo skill .agents/skills/writing-docs/ (SKILL.md, REFERENCE.md, EXAMPLES.md)
  - global skill ~/.pi/agent/skills/writing-code-doc/ (SKILL.md, references/REFERENCE.md)

Scanned shapes, per file:
- .nim carries whole-line ## / /// doc comments, whole-line # maintainer comments, and trailing comments after code (the comment text only)
- .py carries module, class, and function docstrings plus # comments, noqa comments exempt as tooling metadata
- .md carries every line, fenced code blocks included

- code lines, tables (2+ pipe characters), and URL lines are exempt
- diagram lines (box-drawing characters, 2+ arrow markers, mermaid tokens) are exempt
- the Tattletale license header, the skill rule definitions, and this file are exempt

Usage:

    python3 lint_docs.py <files-or-dirs>...
    python3 lint_docs.py --base <commit> <files-or-dirs>...
    python3 lint_docs.py --fix <path>...

When --base <commit> is set (or DOC_LINT_BASE env names the commit), findings
are scoped to the lines a diff from that commit adds, so pre-existing
violations a change did not touch stay out of the report. The diff is read
from the staged index relative to the base commit, falling back to the
working-tree diff when nothing is staged. Without --base, every file is
linted in full.

Output is one finding per line in the `path:line: rule-id: reason` shape,
sorted by path and line.
- exit 0 means clean
- exit 1 means at least one counted finding
- exit 2 means a usage or tool failure
- advisory findings print but never set the exit code

The --fix mode runs the mechanical autofix over every collected file,
writing the fixed text back.
- whole-line comment lines, docstring bodies, and Markdown prose
  paragraphs are the only text the pass rewrites
- code tokens, trailing comments, and command blocks stay untouched
- a transform that would leave any new finding of any rule is skipped,
  the site stays for the LLM pass

| mechanical class            | transform                                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| article-eol, stray-fragment | rewrap the paragraph so no line ends on a dangling article, a bare connective, or a 1-2 word tail after a period |
| single-word-eol             | merge the stub line into the reflow, the last line keeps 3+ words                                                |
| unit-split                  | rewrap when a pure rewrap heals the severed `word,` continuation                                                 |
| colon-inline                | break after the colon, the continuation indented two spaces under the lead, only when the tail rewraps clean     |
| wall-no-air                 | one-sentence blocks merge to 3 or fewer lines; everything else needs judgment                                    |

Leftover findings print tagged, [mechanical] for a class the transform
could not heal and [judgment] for the rest.
- exit 0 in --fix mode means every remaining finding is a judgment class
- exit 1 in --fix mode means at least one mechanical finding remains
"""

import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

PROSE_CAP = 140
SINGLE_WORD_EOL_PREV_MAX = 110
WALL_OF_TEXT_LINES = 10
WALL_NO_AIR_LINES = 4
# A table cell over this many words is a wall of prose in a cell,
# split it into bullets or a diagram.
TABLE_CELL_MAX_WORDS = 30
MODULE_HEADER_MAX_LINES = 8
# Bullet-shape caps. A bullet list holds at most 3 items and one item
# spans at most 3 lines, a 4-item list is banned, longer bullet walls
# belong in a table or diagram.
BULLET_LIST_MAX_ITEMS = 3
BULLET_ITEM_MAX_LINES = 3

ARTICLE_EOL = {"the", "a", "an", "this", "that", "its", "their", "both", "own"}
CONNECTIVE_EOL = {"with", "of", "for", "to", "in", "and", "or", "on", "at",
                  "by", "from", "as", "so", "then", "when"}
ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "cf"}

# Sentence starters that legitimately precede the bare word `newline`.
# Any other Capitalized + `newline` adjacency is escape residue.
ESCAPE_PAIR_EXEMPT = {
    "the", "this", "that", "these", "those", "a", "an", "its", "it", "when",
    "where", "while", "if", "each", "every", "both", "any", "all", "some",
    "one", "two", "how", "why", "what", "which", "who", "whose", "their",
}

DIAGRAM_CHARS = set("│└├┌┐┘┤┬┴─▼▲►◄")

# The bullet lead is a list marker followed by whitespace, optionally indented.
# (a wrapped bullet continuation keeps the lead's indent).
BULLET_RE = re.compile(r"^\s*([-*+•]|\d+[.)])\s")

# Run-command shape for test file headers (test files carry the run command).
RUN_CMD_RE = re.compile(
    r"\bnim test_\w+|^\s*(?:\$|nim\s|python3?\s|uv\s|pytest\b)|(?i:run):\s"
    r"|\btest_\w+ task in config\.nims")

# The test file name shapes differ by language.
# Nim tasks scan test_* and t_* files.
# Python uses test_* and *_test files.
NIM_TEST_FILE_RE = re.compile(r"(?:^|/)(?:test_|t_)[^/]*\.nim$")
PY_TEST_FILE_RE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")

# The missing-contract rule checks these contract markers.
# Labeled sections (Args:, Output:) count through their colon shape.
# The bare contract words count by themselves.
CONTRACT_MARKER_RE = re.compile(
    r"\b(?:args|returns|raises|usage|parameters)\b|\boutput\s*:|"
    r"\b(?:contract|precondition|postcondition|invariants?|expected input|"
    r"input shapes|output shape|lifetime|data flow)\b", re.IGNORECASE)

# Exported Nim declarations the missing-doc rule covers (conservative shapes).
NIM_EXPORTED_CALLABLE_RE = re.compile(
    r"^\s*(?:proc|func|macro|iterator|template|converter)\s+(\w+)\*(?:\s*\[|\s*\()")
NIM_EXPORTED_TYPE_RE = re.compile(
    r"^\s*(?:type\s+)?(\w+)\*\s*=\s*(?:object|ref|distinct)\b")

# The doc-above-type rule. The house doc comment of a type lives inside
# its body, above the fields it describes (the placement object fields
# use), a ## block directly above the declaration does not attach that way.
#
# The bare single-line form after a ## block is flagged too.
TYPE_KEYWORD_RE = re.compile(r"^\s*type\b")
TYPE_DECL_RE = re.compile(
    r"^\s*[A-Za-z_]\w*\*?\s*(?:\[[^\]]*\])?\s*=\s*(?:ref\s+)?"
    r"(?:object|distinct|enum|tuple)\b")
SECTION_END_RE = re.compile(
    r"^\s*(?:proc|func|macro|iterator|template|converter|const|let|var|import|"
    r"export|include)\b")

# Temporal and history narration (the narration rule).
# Each entry carries the pattern plus a one-line hint.
#
# Patterns stay narrow, only phrases naming a past or pending state
# the reader cannot observe.
NARRATION = [
    (r"\b(?:currently|presently|formerly|previously)\b",
     "temporal status marker (state the invariant in present tense)"),
    (r"\bas of\b", "temporal marker (state the fact without a date anchor)"),
    (r"\bat the time of\b", "temporal marker (state the fact, not the moment)"),
    (r"\bfor now\b", "temporal marker (state the invariant instead)"),
    (r"\bnow that\b", "temporal connective (state the resulting state instead)"),
    (r"\bno longer\b", "past-state narration (describe the code as it is now)"),
    (r"\bused to\b", "past-state narration (describe the code as it is now)"),
    (r"\bwas (?:buggy|broken|wrong|failing|incorrect)\b",
     "past-state narration (state the invariant instead)"),
    (r"\broot cause\b", "bug postmortem label (state the invariant instead)"),
    (r"\bnow (?:fixed|works|green)\b", "transient status marker"),
    (r"\bwent (?:green|red)\b", "transient test-status marker"),
    (r"\bturns? (?:green|red)\b", "transient test-status marker"),
    (r"\bfails now\b|\bexpected to fail\b",
     "transient test-status marker (state the invariant instead)"),
    (r"\b(?:fixed|resolved) (?:by|in|with)\b",
     "history narration (state the behavior, not the fix)"),
    (r"\bworkaround for\b", "workaround narration (state the behavior instead)"),
    (r"\bonce\b[^.;]*\b(?:lands?|merges?|ships?|arrives?)\b",
     "future-promise narration (name the dependency by path or name instead)"),
    (r"\bhas been (?:fixed|moved|renamed|removed|deleted|replaced)\b",
     "history narration (describe the code as it is now)"),
    (r"\bwill be (?:removed|replaced|deleted)\b",
     "future-promise narration (use a TODO naming the missing condition)"),
]

# Pipeline artifact references (the artifact-ref rule).
# Finding IDs, iteration labels, and ruling numbers name
# nothing in a fresh clone.
ARTIFACT = [
    (r"\b(?:SLOP|QA|RID|INV|REQ|HIDN|SPEC|GOAL|ESC|DEV|TICKET)-\d+\b",
     "pipeline artifact reference"),
    (r"\bBUG-[A-Z]*\d+[A-Z]*\b", "pipeline artifact reference"),
    (r"\biter-\d+\b", "iteration label reference"),
    (r"\bJ\d{2,4}\b", "ruling-number reference (state the fact instead)"),
]

# Narration of the how over the what (the how-narration rule, advisory).
HOW_NARRATION = [
    (r"^\s*This (?:function|method|class|module|procedure|proc|loop|test) ",
     "narrates the code instead of stating the contract"),
    (r"^\s*Here,? we\b", "narrates the author's walk-through"),
    (r"^\s*We (?:iterate|walk|loop|increment|start by|first)\b",
     "narrates the author's walk-through"),
    (r"\bmake sure (?:that|to)\b",
     "instruction leaking (state the precondition instead)"),
    (r"\bremember to\b", "instruction leaking (state the requirement instead)"),
    (r"\bdon't forget\b", "instruction leaking (state the requirement instead)"),
]

# Banned vocabulary extends the EXAMPLES.md hard blocklist with operator extensions.
#
# Each entry carries the pattern, the optional exemption predicate,
# and the replacement hint.
#
# The predicate reads the lowercased line with backtick spans stripped.
# The hint text carries no banned word.
# Finding messages quote the runtime match instead.
def _probe_exempt(line):
    """Exemptions for the record-field vocabulary.

    - Args:
      the lowercased prose line under check
    - Returns:
      True in the schema contexts listed below
    - Contract:
      the fingerprint precedent applies
      format names are acceptable vocabulary
      the test-file sense stays banned

    Schema contexts kept legal:

    - `probe_*` schema keys
    - the field listing, `probe fields`
    - boundary-element mentions, `(probe)` parenthesized
    """
    return bool(re.search(r"probe_|\(probe\)|probe fields|probe elements", line))


BANNED = [
    (r"\blegacy\b|\bhistorically\b|\bpreviously\b|\bformerly\b"
     r"|\boutdated\b|\bobsolete\b|\bcurrently\b"
     r"|\bno longer\b|\bused to \b|\bas of \b",
     None, "historical and temporal prose is banned, state the present contract "
           "(the schema before X, absent keys, the pre-005 frames)"),
    (r"\bpin\b|\bpins\b|\bpinned\b|\bpinning\b",
     None, "use verified against, checked by, locked"),
    (r"\bgate\b|\bgates\b|\bgating\b",
     lambda l: bool(re.search(r"sigmoid|gated|gate\.weight|_gate|gate_|gating (network|mechanism|layer)|swiglu|\bglu\b|\bweight", l)),
     "use test, check, or assert (model-architecture names exempt)"),
    (r"\bcensus\b|\bcensuses\b|\bcensusing\b",
     None, "use counts or per-element counts"),
    (r"\bdraw\b|\bdraws\b", None, "use read, take, or use"),
    (r"\bride\b|\brides\b|\briding\b|\bridden\b|\brode\b", None,
     "use sit on, carry, or restate the mechanism"),
    (r"\bbite\b|\bbites\b", None, "use chunk, step, or case"),
    (r"\bmissions?\b", None, "use the module's real name or path"),
    (r"\bdigests?\b",
     lambda l: bool(re.search(r"\bsha|hash|checksum|blake|md5", l)),
     "use summary or report (the cryptographic sense stays exempt)"),
    (r"\bmutations?\b", None, "use change or variation"),
    (r"\bconvict[s]?\b|\bacquit[s]?\b", None,
     "state what the comparison shows, use localizes or rules out"),
    (r"\breceipts?\b", None,
     "cite the command and its output that prove the claim"),
    (r"\bpostures?\b", None,
     "use build variant, configuration, or name the flags"),
    (r"\brungs?\b", None,
     "name the ladder tier, 00 codec, 01 per-op, 02 chain, 03 forward, 04 decode"),
    (r"\bsubstrates?\b", None,
     "use base, foundation, or name the component"),
    (r"\bRED\b|\bGREEN\b", None,
     "state the invariant in present tense"),
    (r"\boracles?\b", None, "use reference implementation"),
    (r"\bprobes?\b", _probe_exempt,
     "use test (the record field name `probe` and `probe_*` schema keys stay)"),
    (r"\bdeviation classes?\b", None, "describe the actual difference"),
    (r"\bload[- ]bearing\b", None, "use essential, critical, or necessary"),
    (r"\bseams?\b", None, "use boundary, interface, or edge"),
    (r"\bphysics[- ]bearing\b|\bphysics\b",
     None, "use honest rounding, drift behavior, or value-bearing"),
    (r"\bcommitted (?:bytes|blobs?|blob hashes)\b",
     None, "use recorded inputs, recorded files, or recorded checksums"),
    (r"\btail[- ]mass\b", None, "use tail probability"),
    (r"\barms?\b", None, "use variant or rephrase"),
    (r"\brails?\b", None, "use reference or reference path"),
    (r"\bbatter(?:y|ies)\b", None, "use checks or suite"),
    (r"\bwaves?\b", None, "use pass or work"),
    (r"\bdonors?\b", None, "use recorded family or recorded source"),
    (r"\btripwires?\b", None, "use guard or check"),
    (r"\btwin\b", None, "use the recorded row or the paired row"),
    (r"\blaws?\b", None,
     "use rule, rule set, contract, or recorded ruling"),
    (r"\bincumbents?\b", None,
     "use the recorded frame, the committed frame, or the existing recording"),
    (r"\benvelopes?\b", None,
     "state the bound directly (the band name or the inequality)"),
]

LICENSE_SHAPE = re.compile(
    r"^(Tattletale|Copyright \(c\)|Licensed and distributed|"
    r"\* (MIT|Apache)|at your option|This file may not be copied)")

# Embedded code-sample shapes inside doc comments:
# exempt from the line-break and parenthesis rules.
CODEISH = re.compile(
    r"^(?:var |let |const |func |proc |iterator |macro |template |type |import |"
    r"include |from |export |echo |return |if |elif |else:|for |while |case |"
    r"of |discard |raise |result\s*=|\w+\s*=\s*\S|#!)")

ESCAPE_SEQ = re.compile(r"\\(?:newline\b|[nrt0]\b|x[0-9A-Fa-f]{2}\b|u[0-9A-Fa-f]{4}\b)")
ESCAPE_PAIR = re.compile(r"\b([A-Z][a-z]+)\s+(newline)\b")

TRAILING_DOC = re.compile(r"^(?P<code>.+?)\s+##(?P<rest>.*)$")
TRAILING_HASH = re.compile(r"^(?P<code>.+?)\s+#(?!#)(?P<rest>.*)$")


class Rule:
    """Registry record carrying the rule id, the counted flag, and the trigger text."""
    __slots__ = ("rule", "counted", "trigger")

    def __init__(self, rule, counted, trigger):
        self.rule, self.counted, self.trigger = rule, counted, trigger


RULES = {
    "banned-vocab": Rule("banned-vocab", True,
                         "a blocklist word (EXAMPLES.md, plus operator-extended entries)"),
    "the-opener": Rule("the-opener", True,
                       "a doc comment, maintainer comment or heading opens with the article \"The\""),
    "semicolon": Rule("semicolon", True, "a semicolon in prose"),
    "em-dash": Rule("em-dash", True, "an em-dash or en-dash in prose"),
    "line-length": Rule("line-length", True, "a prose line over 140 characters"),
    "article-eol": Rule("article-eol", True,
                        "a line ends on a dangling article or a stranded possessive"),
    "stray-fragment": Rule("stray-fragment", True,
                           "a line ends on a bare connective or a fragment after the period"),
    "colon-break": Rule("colon-break", True,
                        "a colon orphaned at line start or split from its lead phrase"),
    "colon-inline": Rule("colon-inline", True,
                         "a prose colon followed by prose on the same line"),
    "unit-split": Rule("unit-split", True,
                       "a line opens on a severed one-word continuation"),
    "paren-split": Rule("paren-split", True, "a line ends inside an open parenthesis"),
    "single-word-eol": Rule("single-word-eol", True,
                            "a 1-2 word stub line with reflow room on the previous line"),
    "doc-above-type": Rule("doc-above-type", True,
                           "a ## block sits directly above a type declaration"),
    "bullet-list-length": Rule("bullet-list-length", True,
                               "a bullet list with 4 or more items"),
    "bullet-item-length": Rule("bullet-item-length", True,
                               "a single bullet item spanning 4 or more lines"),
    "table-separator": Rule("table-separator", True,
                            "a table with no |---| separator row after the header"),
    "table-mispadding": Rule("table-mispadding", True,
                              "a table row with a different cell count than the header, or a doc table row with an unpadded cell"),
    "table-cell-wall": Rule("table-cell-wall", True,
                             "a table cell over TABLE_CELL_MAX_WORDS words (split the cell into bullets or a diagram)"),
    "table-alignment": Rule("table-alignment", True,
                             "a table whose pipe separators do not line up (pad each cell to the column width)"),
    "escape-noise": Rule("escape-noise", True, "an escape sequence used as a prose word"),
    "narration": Rule("narration", True,
                      "temporal or history narration: currently, as of, once X lands"),
    "artifact-ref": Rule("artifact-ref", True,
                         "a pipeline artifact ID: SLOP-002, QA-004, iter-3"),
    "wall-of-text": Rule("wall-of-text", False,
                         "10+ consecutive prose lines with no bullet, table, or diagram"),
    "wall-no-air": Rule("wall-no-air", True,
                        "the longest consecutive airless prose stretch reaches WALL_NO_AIR_LINES lines"),
    "missing-diagram": Rule("missing-diagram", False,
                            "a multi-step flow described in prose or bullets with no diagram"),
    "how-narration": Rule("how-narration", False,
                          "a doc narrates the how: \"This function...\", \"we iterate\", \"make sure\""),
    "module-header-length": Rule("module-header-length", False,
                                 "a module header past 8 tight prose lines"),
    "test-header-command": Rule("test-header-command", True,
                                "a test file header with no run command line"),
    "missing-doc": Rule("missing-doc", True,
                        "a public item with no doc comment (exported Nim proc or type, module-level Python def or class)"),
    "doc-above-proc": Rule("doc-above-proc", True,
                           "a ## block directly above a proc or func declaration, the house doc comment is the first body line"),
    "missing-contract": Rule("missing-contract", False,
                             "a multi-line function doc with no contract marker (Args, Returns, Contract, Invariant)"),
    "sig-wrap": Rule("sig-wrap", True,
                     "a proc or func signature wrapped across lines while the joined form fits a 140-char line"),
    "except-rewrap": Rule("except-rewrap", True,
                          "an except clause re-raises the caught exception (rewrap; handle it or let it propagate)"),
    "design-narration": Rule("design-narration", True,
                             "a doc or maintainer comment justifying the design choice instead of stating the contract (because, instead of, rather than, X and not Y, which is why, declared ahead)"),
    "section-separator": Rule("section-separator", True,
                              "a whole-line # comment built from dashes (a layout-position marker; keep the title line, drop the rule)"),
    "try-block": Rule("try-block", True,
                      "a try/except or try/finally block catching exceptions as control flow outside the libtorch C++ boundary and tests folders"),
}


class Finding:
    """One linter finding carrying the file path, line number, rule id, reason, and advisory flag."""
    __slots__ = ("path", "line", "rule", "reason", "warning")

    def __init__(self, path, line, rule, reason, warning=False):
        self.path, self.line, self.rule = path, line, rule
        self.reason, self.warning = reason, warning


def strip_backticks(text):
    """Removes backticked spans so code tokens inside prose never skew the checks."""
    return re.sub(r"`[^`]*`", " ", text)


def is_diagram(text):
    """Returns True for a spatially-structured diagram line.

    A line is a diagram when it carries box-drawing characters, a mermaid
    diagram token, or 2+ arrow markers. A single `-->` or `→` in prose is
    an arrow for readability, not a diagram, so a line like
    `s --> t = base(s) xor b` stays prose under the rules.
    """
    if any(ch in DIAGRAM_CHARS for ch in text):
        return True
    if (text.count("→") + text.count("←")
            + text.count("-->") + text.count("──") >= 2):
        return True
    if re.match(r"^(?:graph|flowchart|sequenceDiagram|stateDiagram|erDiagram|"
                r"classDiagram)\b", text.strip()):
        return True
    return False


def is_table(text):
    """Returns True for table rows holding two or more pipe characters.

    A real table row opens on a pipe (the comment marker the extractors
    already strip), so a prose line that merely carries two pipes mid-line
    (e.g. `use a | b | c`) stays prose and is never mistaken for a table.
    """
    return text.count("|") >= 2 and text.lstrip().startswith("|")


def _is_separator_row(text):
    """Returns True for a table separator row, every cell a run of 2+
    dashes with optional alignment colons, e.g. `|---|---|`."""
    cells = [c.strip() for c in text.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _cell_count(text):
    """Counts the cells in a pipe-delimited table row."""
    body = text.strip().strip("|")
    return 0 if not body else len(body.split("|"))


def _cell_word_count(text):
    """Returns the word count of a table cell, backticked spans stripped."""
    return len(words(strip_backticks(text)))


def _table_cells_padded(text):
    """Returns True when every cell in a table row carries a leading and
    trailing space (e.g. `| a | b |`), False when a cell is unpadded."""
    body = text.strip().strip("|")
    if not body:
        return True
    return all(c.startswith(" ") and c.endswith(" ") for c in body.split("|"))


def _separator_cells(cells):
    """Returns True when every stripped cell is a separator cell, a run
    of 2+ dashes with optional alignment colons (e.g. `---`, `:---:`, `---:`)."""
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _split_table_row(line):
    """Splits a table row into its comment prefix and its content cells.

    - the prefix is everything before the first pipe, e.g. `## ` or the empty
      string when the content carries no comment marker
    - cells are split on the pipe-space-pipe delimiter, so a literal pipe
      inside a cell does not split the cell
    - a trailing empty cell from the closing pipe is dropped, so the count
      never includes the closing pipe as a field
    """
    prefix, sep, tail = line.partition("|")
    if not sep:
        return prefix, []
    body = tail
    if body.startswith(" "):
        body = body[1:]
    if body.endswith(" |"):
        body = body[:-2]
    elif body.endswith("|"):
        body = body[:-1]
    if not body.strip():
        return prefix, []
    return prefix, [c.strip() for c in body.split(" | ")]


def _table_row_padded(line):
    """Returns True when a table row carries a space around every cell.

    A padded row opens with a space after the first pipe and splits into two
    or more cells. An unpadded row splits into a single fused cell, so it
    never qualifies here.
    """
    return len(_split_table_row(line)[1]) >= 2


def _render_separator_cell(cell, width):
    """Returns one separator cell filled with dashes to the column width.

    A `:---:` cell at width 19 renders 17 dashes between the colons, keeping
    the total cell width at the column width. A column narrower than 3 keeps
    at least 3 dashes.
    """
    m = re.fullmatch(r"(:?)(-+)(:?)", cell)
    if not m:
        return cell.ljust(width)
    left, _dashes, right = m.groups()
    n = max(width - len(left) - len(right), 3)
    return left + "-" * n + right


def _render_aligned_row(prefix, cells, widths):
    """Returns one table row with each cell padded to its column width.

    A separator row fills its cells with dashes to the column width.
    A content row left-justifies each cell to the width, so the pipe
    separators line up across the whole table.
    """
    parts = []
    sep = _separator_cells(cells)
    for idx, cell in enumerate(cells):
        if sep:
            parts.append(_render_separator_cell(cell, widths[idx]))
        else:
            parts.append(cell.ljust(widths[idx]))
    return prefix + "| " + " | ".join(parts) + " |"


def is_url_line(text):
    """Returns True for lines carrying a URL (always exempt)."""
    return bool(re.search(r"https?://|\bwww\.", text))


def eol_token(text):
    """Returns the line's final word with punctuation stripped and lowercased,
    or None when the tail is compound, empty, or not a plain word."""
    tail = text.rstrip()
    if not tail:
        return None
    last = re.split(r"\s+", tail)[-1]
    if "/" in last or "`" in last:
        return None
    last = last.strip(".,;:()'\"*")
    if not last or not last.isalpha():
        return None
    return last.lower()


def words(text):
    """Splits prose into its whitespace-separated words."""
    return [w for w in re.split(r"\s+", text.strip()) if w]


def code_like(text):
    """Returns True for embedded code-sample lines (exempt from prose rules)."""
    return bool(CODEISH.match(text))


def comment_indent(body):
    """Returns the leading-whitespace width of a comment body, the wrap
    indent that marks a bullet continuation line. A blank body reads 0."""
    stripped = body.lstrip()
    if not stripped:
        return 0
    return len(body) - len(stripped)


def nim_prose_lines(text):
    """Extracts prose from a Nim source file.

    Returns (entries, None).
    - each entry carries the line number, the content, and the kind
    - kind reads `doc` for ## and /// lines, `hash` for maintainer # lines
    - the wrap indent and the trailing flag mark comments anchored to code items

    License-header lines and empty `#` lines are dropped, a `##`-only line
    keeps its paragraph break as air.
    """
    out = []
    in_block_comment = False
    raws = text.splitlines()

    def banner_rule(s):
        core = s.lstrip("#").strip()
        return bool(core) and set(core) <= {"#", " "} or \
            bool(core) and set(core) <= {"-", " "}

    for i, raw in enumerate(raws, 1):
        line = raw.rstrip()
        s = line.strip()
        if in_block_comment:
            if "]#" in s:
                in_block_comment = False
            continue
        if s.startswith("#["):
            in_block_comment = True
            continue
        if s.startswith("##"):
            out.append((i, s[2:].strip(), "doc", comment_indent(s[2:]), False))
            continue
        if s.startswith("///"):
            out.append((i, s[3:].strip(), "doc", comment_indent(s[3:]), False))
            continue
        if s.startswith("#"):
            if s == "#" or LICENSE_SHAPE.match(s.lstrip("#").strip()):
                continue
            if banner_rule(s):
                continue
            body = s.lstrip("#")
            core = body.strip()
            # the section banners carry centered titles: a title sits
            # between two rule lines, a subsection title owns the dash
            # rule below it, neither is prose
            if core:
                prev = raws[i - 2].strip() if i >= 2 else ""
                nxt = raws[i].strip() if i < len(raws) else ""
                if prev.startswith("#") and nxt.startswith("#") \
                        and banner_rule(prev) and banner_rule(nxt):
                    continue
                if nxt.startswith("#") and banner_rule(nxt) \
                        and "-" in nxt:
                    continue
            out.append((i, body.strip(), "hash", comment_indent(body), False))
            continue
        m = TRAILING_DOC.match(line)
        if m and m.group("code").count('"') % 2 == 0:
            out.append((i, m.group("rest").strip(), "doc", 0, True))
            continue
        m = TRAILING_HASH.match(line)
        if m and m.group("code").count('"') % 2 == 0:
            rest = m.group("rest").strip()
            if rest:
                out.append((i, rest, "hash", 0, True))
    return out, None


_PY_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef,
                        ast.AsyncFunctionDef)
_PY_OPEN_QUOTE_RE = re.compile(r"^([A-Za-z]*)('{1,3}|\"{1,3})")


def _strip_string_frame(body, quote):
    """Removes the closing quote frame from a docstring body line."""
    body = body.rstrip()
    if quote:
        while body.endswith(quote[-1]):
            cut = len(quote) if body.endswith(quote) else 1
            body = body[:-cut].rstrip()
            if not body.endswith(quote[-1]):
                break
    return body


def py_prose_lines(text):
    """Extracts prose from a Python source file.

    Returns (entries, meta). Each entry is a tuple of line number, content,
    kind (`doc` for docstring lines, `hash` for comment lines), wrap indent,
    and a trailing flag marking comments anchored to code items.

    Meta maps the module docstring lines to `module_header`, the class
    and function docstring pairs into `func_docs`, and the raw shapes
    the autofix re-emits from into `comment_toks` plus `docstring_spans`.
    """
    lines = text.splitlines()
    entries = []
    module_header = []
    func_docs = []
    docstring_meta = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, _PY_DOCSTRING_OWNERS):
                continue
            first = node.body[0] if node.body else None
            if (first is None or not isinstance(first, ast.Expr)
                    or not isinstance(first.value, ast.Constant)
                    or not isinstance(first.value.value, str)):
                continue
            expr = first.value
            quote_m = _PY_OPEN_QUOTE_RE.match(lines[expr.lineno - 1][expr.col_offset:])
            quote = quote_m.group(2) if quote_m else ""
            docstring_meta.append(
                (expr.lineno, expr.end_lineno, expr.col_offset, quote,
                 isinstance(node, ast.Module)))
    docstring_meta.sort()
    for start, end, base, quote, is_module in docstring_meta:
        span_entries = []
        for n in range(start, end + 1):
            raw = lines[n - 1].rstrip()
            if n == start:
                body = raw[base:]
                body = re.sub(r"^[A-Za-z]*", "", body, count=1)
                if quote and body.startswith(quote):
                    body = body[len(quote):]
            else:
                lead = len(raw) - len(raw.lstrip())
                body = raw[min(lead, base):]
            if n == end:
                body = _strip_string_frame(body, quote)
            indent = comment_indent(body)
            content = body.strip()
            span_entries.append((n, content, "doc", indent, False))
            if content:
                if is_module:
                    module_header.append((n, content))
        entries.extend(span_entries)
        if not is_module:
            func_docs.append((start, [e[1] for e in span_entries if e[1]]))
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        comment_toks = [tok for tok in tokens if tok.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        comment_toks = []
    for tok in comment_toks:
        n = tok.start[0]
        before = lines[n - 1][:tok.start[1]]
        if not before.strip():
            body = tok.string.lstrip("#")
            if n == 1 and body.startswith("!"):
                continue
            if NOQA_RE.match(body):
                continue
            entries.append((n, body.strip(), "hash", comment_indent(body), False))
        else:
            rest = tok.string.lstrip("#").strip()
            if rest and not NOQA_RE.match(rest):
                entries.append((n, rest, "hash", 0, True))
    entries.sort(key=lambda e: e[0])
    return entries, {"module_header": module_header, "func_docs": func_docs,
                     "comment_toks": [(t.start[0], t.start[1], t.string)
                                      for t in comment_toks],
                     "docstring_spans": list(docstring_meta)}


def md_prose_lines(text):
    """Extracts prose from a Markdown file, every line and fenced code
    blocks included, minus frontmatter, tables, URL lines, diagram lines.

    Contract:
    - fenced blocks are scanned the way code is scanned
    - headings carry kind `heading` with the heading text as content
    - every entry carries the raw line's indent width
    """
    out = []
    lines = text.splitlines()
    i = 0
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            i += 1
        i += 1
    fence = False
    for n in range(i, len(lines)):
        raw = lines[n]
        s = raw.strip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
            continue
        if not s:
            continue
        if is_url_line(s) or is_table(s) or is_diagram(s):
            continue
        if re.match(r"^(?:=+|-{3,}|_{3,}|\*{3,})$", s):
            continue
        if s.startswith("<!--") or s.startswith("<div"):
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            out.append((n + 1, m.group(2).strip(), "heading", 0, False))
            continue
        out.append((n + 1, s, "fence" if fence else "prose",
                    len(raw) - len(raw.lstrip()), False))
    return out, None


def extract(path, text):
    """Dispatches on the file suffix to the Nim, Python, and Markdown extractors."""
    if path.suffix == ".nim":
        return nim_prose_lines(text)
    if path.suffix == ".py":
        return py_prose_lines(text)
    if path.suffix == ".md":
        return md_prose_lines(text)
    return [], None


def structural(c):
    """Returns True for air lines holding bullets, tables, diagrams, or URL lines."""
    return is_table(c) or is_diagram(c) or is_url_line(c) or bool(BULLET_RE.match(c))


def flush_wall(path, run, findings):
    """Fires the advisory wall-of-text rule on a prose run longer than
    WALL_OF_TEXT_LINES lines that holds no bullet, table, or diagram line."""
    if len(run) > WALL_OF_TEXT_LINES:
        findings.append(Finding(
            path, run[0][0], "wall-of-text",
            "%d consecutive prose lines with no bullet, table, or diagram" % len(run),
            warning=True))


def flush_wall_no_air(path, run, findings):
    """Fires the counted wall-no-air rule on the longest consecutive
    airless prose stretch inside a sub-block.

    A stretch is a run of consecutive prose lines with no bullet, table,
    diagram, or URL line between them. Fire when that stretch reaches
    WALL_NO_AIR_LINES lines, so a dense paragraph next to a bullet or
    table is still caught instead of being excused by the nearby air.
    """
    best_start = None
    best_len = 0
    cur_start = None
    cur_len = 0
    for n, air in run:
        if air:
            cur_start = None
            cur_len = 0
            continue
        if cur_start is None:
            cur_start = n
        cur_len += 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
    if best_len >= WALL_NO_AIR_LINES and best_start is not None:
        findings.append(Finding(
            path, best_start, "wall-no-air",
            "prose stretch of %d lines has no bullet, table, or diagram "
            "between them: add air (bullet the list-able content, "
            "split the paragraph)" % best_len))


def _compile_table(table, ignore_case):
    """Precompiles a pattern table with the table's case sensitivity."""
    flags = re.IGNORECASE if ignore_case else 0
    return [(re.compile(pattern, flags), hint) for pattern, hint in table]


NARRATION_RE = _compile_table(NARRATION, True)
ARTIFACT_RE = _compile_table(ARTIFACT, False)
HOW_NARRATION_RE = _compile_table(HOW_NARRATION, True)


def _pattern_findings(path, n, c, table, rule, findings, warning=False):
    """Fires a pattern-table rule once per matching pattern on the line."""
    bare = strip_backticks(c)
    for pattern, hint in table:
        m = pattern.search(bare)
        if not m:
            continue
        findings.append(Finding(
            path, n, rule, "%s: '%s' (%s)" % (hint, m.group(0), hint),
            warning=warning))


def check_line(path, n, c, kind, is_nim, prev_text, findings):
    """Runs the per-line rules over one prose line."""
    if not c:
        return
    bare = strip_backticks(c)
    if code_like(c) or is_table(c) or is_diagram(c) or is_url_line(c):
        return
    # The title flag fires on a short colon-terminated line.
    # Every comma-separated title segment opens on a noun phrase,
    # including the lowercase article form.
    if c.endswith(":"):
        segs = [s.strip() for s in bare.split(",") if s.strip()]
        if segs and all(len(s.split()) <= 8 for s in segs) and any(
                s.split()[0].lower() == "the" for s in segs):
            findings.append(Finding(
                path, n, "the-opener",
                "title opens with the (open with a noun phrase)"))
    if kind == "fence":
        # Fenced content is code-with-layout.
        # - the vocabulary rules apply
        # - the prose-structure rules stay off code lines
        # - no fence carries a general exemption
        #
        # REFERENCE.md alone holds the real-life bad examples.
        # A fully quoted line there cites banned prose on purpose.
        # The exact wording is the evidence, so it stays exempt.
        s = c.strip()
        if (str(path).endswith("REFERENCE.md") and len(s) > 1
                and s.startswith('"') and s.endswith('"')):
            return
        if "—" in c or "–" in c:
            findings.append(Finding(path, n, "em-dash", "em-dash in prose"))
        low = bare.lower()
        for pattern, exempt, hint in BANNED:
            hay = c if pattern.startswith(r"\bRED") else low
            m = re.search(pattern, hay)
            if not m:
                continue
            if exempt and exempt(low):
                continue
            findings.append(Finding(
                path, n, "banned-vocab",
                "banned vocabulary: '%s' (%s)" % (m.group(0), hint)))
        return
    # A prose colon ends its line under the colon-inline rule.
    # What it introduces goes on the next lines, never the same line.
    # URLs keep their scheme colon.
    cm = re.search(r":\s+\S", bare)
    if cm and "://" not in bare[:cm.start() + 2]:
        findings.append(Finding(
            path, n, "colon-inline",
            "a prose colon is followed by prose on the same line "
            "(move what it introduces to the next lines)"))
    # The unit-split rule fires when a severed continuation opens the line.
    # The previous line holds the subject, this line holds "apply,".
    if re.match(r"^[a-z]+,", bare.strip()) and prev_text.get(n - 1):
        findings.append(Finding(
            path, n, "unit-split",
            "the line opens on a severed one-word continuation "
            "(keep the semantic unit on one line)"))
    if len(c) > PROSE_CAP:
        findings.append(Finding(
            path, n, "line-length",
            "prose line is %d chars, cap is %d" % (len(c), PROSE_CAP)))
    if ";" in bare:
        findings.append(Finding(path, n, "semicolon", "semicolon in prose"))
    if "—" in c or "–" in c:
        findings.append(Finding(path, n, "em-dash", "em-dash in prose"))
    low = bare.lower()
    for pattern, exempt, hint in BANNED:
        hay = c if pattern.startswith(r"\bRED") else low
        m = re.search(pattern, hay)
        if not m:
            continue
        if exempt and exempt(low):
            continue
        findings.append(Finding(
            path, n, "banned-vocab",
            "banned vocabulary: '%s' (%s)" % (m.group(0), hint)))
    _pattern_findings(path, n, c, NARRATION_RE, "narration", findings)
    _pattern_findings(path, n, c, ARTIFACT_RE, "artifact-ref", findings)
    _pattern_findings(path, n, c, HOW_NARRATION_RE, "how-narration",
                      findings, warning=True)
    if ESCAPE_SEQ.search(bare):
        findings.append(Finding(
            path, n, "escape-noise",
            "escape sequence used as a prose word (backtick it or spell the meaning)"))
    m = ESCAPE_PAIR.search(bare)
    if m and m.group(1).lower() not in ESCAPE_PAIR_EXEMPT:
        findings.append(Finding(
            path, n, "escape-noise",
            "escape residue: bare %s glued to an identifier "
            "(backtick the code token or spell the meaning)" % m.group(2)))
    endtxt = re.sub(r"`[^`]*`", " XcodeX ", c)
    tok = eol_token(endtxt)
    if tok == "xcodex":
        # The line ends on a backticked code span. The verdict moves
        # to the word before the span, where an article still dangles.
        prev = re.sub(r"XcodeX\s*$", "", endtxt.rstrip()).rstrip()
        ptok = eol_token(prev)
        if ptok in ARTICLE_EOL:
            findings.append(Finding(
                path, n, "article-eol",
                "line ends on the dangling article `%s` before a code span" % ptok))
        tok = None
    if tok:
        if tok in ARTICLE_EOL:
            findings.append(Finding(
                path, n, "article-eol",
                "line ends on the dangling article `%s`" % tok))
        elif tok.endswith("'s"):
            findings.append(
                Finding(path, n, "article-eol", "line ends on a stranded possessive"))
        elif tok in CONNECTIVE_EOL:
            findings.append(Finding(
                path, n, "stray-fragment",
                "line ends on the bare connective `%s`" % tok))
    m = re.search(r"(?<![.\d])\.\s+(\S+(?:\s+\S+)?)$", bare)
    if m:
        lead = m.group(1).split()[0].strip(".,;:()'\"*").lower()
        if lead not in ABBREVIATIONS and "`" not in m.group(1):
            findings.append(Finding(
                path, n, "stray-fragment",
                "sentence-final period followed by a dangling fragment at line end"))
    if re.match(r"^:(?:\s|$)", bare):
        findings.append(Finding(
            path, n, "colon-break",
            "colon orphaned at line start (the colon belongs on the previous line)"))
    else:
        m = re.match(r"^(\w{1,15}):\s+\S", bare)
        if (m and m.group(1).islower() and not BULLET_RE.match(c)):
            prev = prev_text.get(n - 1)
            if (prev is not None and prev.strip()
                    and not re.search(r"[.:;]\s*$", prev)):
                findings.append(Finding(
                    path, n, "colon-break",
                    "single-word colon lead continues the previous sentence "
                    "(keep the colon with its lead phrase)"))
    if c.count("(") > c.count(")"):
        findings.append(Finding(
            path, n, "paren-split",
            "line ends inside an open parenthesis (keep the unit on one line "
            "or end the line before the paren)"))


def _bullet_continuation(entry, lead_indent):
    """Returns True for a list-run member line that is no bullet lead.

    Contract, the item continues when:

    - the line is no fence, heading, empty line, or trailing comment
    - the line indents past the item's lead line (Nim `##` doc lines
      carry indent 1 flush, so the lead's own width is the floor)

    A line at or left of the lead indent ends the item.
    """
    _n, c, kind, indent, trailing = entry
    if not c or trailing or kind in ("fence", "heading"):
        return False
    return indent > lead_indent


def check_bullets(path, block, findings):
    """Fires the bullet-shape rules over one block of prose entries.

    Threshold contract, a bullet list holds at most 3 items and one item
    spans at most 3 lines, a 4-item list is banned. Longer bullet walls
    belong in a diagram, a table, or split lists.

    A list is a maximal run of entries where every member is a bullet
    lead or a continuation indented past its item's lead line.

    - every lead line counts as an item, nested leads included
    - a nested lead ends the item above it
    - a run never crosses an empty line, a heading, a fence, a line at
      or left of the lead indent, or a trailing comment
    - adjacent lists separated by air or prose stay separate runs
    """
    lead_idx = {}
    for i, (_n, c, kind, indent, trailing) in enumerate(block):
        if kind in ("doc", "hash", "prose") and c and not trailing \
                and BULLET_RE.match(c):
            lead_idx[i] = indent
    if not lead_idx:
        return
    i = 0
    while i < len(block):
        if i not in lead_idx:
            i += 1
            continue
        spans = []
        j = i
        while j < len(block):
            if j in lead_idx:
                k = j + 1
                while k < len(block) and k not in lead_idx \
                        and _bullet_continuation(block[k], lead_idx[j]):
                    k += 1
                spans.append((j, k - j))
                j = k
                continue
            break
        if len(spans) > BULLET_LIST_MAX_ITEMS:
            findings.append(Finding(
                path, block[i][0], "bullet-list-length",
                "bullet list holds %d items, cap is %d (split the list, "
                "or move the content into a table or diagram)"
                % (len(spans), BULLET_LIST_MAX_ITEMS)))
        for lead, span in spans:
            if span > BULLET_ITEM_MAX_LINES:
                findings.append(Finding(
                    path, block[lead][0], "bullet-item-length",
                    "bullet item spans %d lines, cap is %d (wrap tighter, "
                    "or move the content into a table or diagram)"
                    % (span, BULLET_ITEM_MAX_LINES)))
        i = j + 1


def check_tables(path, block, findings):
    """Fires the table-separator, table-mispadding, and table-alignment
    rules over a run of pipe-delimited rows.

    A table is a run of 2+ consecutive pipe-delimited rows. A table that
    carries no separator row after its header is flagged, since the
    header and body cannot be told apart. A doc-comment table whose cells
    carry no leading/trailing space is flagged as mispadded. A table whose
    rows pad each cell to a different column width is flagged so the pipe
    separators line up across the whole table.
    """
    i = 0
    n = len(block)
    while i < n:
        if not is_table(block[i][1]):
            i += 1
            continue
        run = []
        j = i
        while j < n and is_table(block[j][1]):
            run.append(block[j])
            j += 1
        if len(run) >= 2:
            if not any(_is_separator_row(row[1]) for row in run):
                findings.append(Finding(
                    path, run[0][0], "table-separator",
                    "table carries no separator row after the header "
                    "(add a |---| row between the header and the body)"))
            header_cells = _cell_count(run[0][1])
            if header_cells:
                for row in run[1:]:
                    if _is_separator_row(row[1]):
                        continue
                    n_cells = _cell_count(row[1])
                    if n_cells != header_cells:
                        findings.append(Finding(
                            path, row[0], "table-mispadding",
                            "table row has %d cells, header has %d "
                            "(align the row to the header columns)"
                            % (n_cells, header_cells)))
            if any(row[2] == "doc" for row in run):
                for row in run:
                    if not _table_cells_padded(row[1]):
                        findings.append(Finding(
                            path, row[0], "table-mispadding",
                            "table row is not padded (give each cell a "
                            "leading and trailing space, e.g. `| a | b |`)"))
            for row in run:
                if _is_separator_row(row[1]):
                    continue
                cells = [c.strip() for c in row[1].strip().strip("|").split("|")]
                for cell in cells:
                    wc = _cell_word_count(cell)
                    if wc > TABLE_CELL_MAX_WORDS:
                        findings.append(Finding(
                            path, row[0], "table-cell-wall",
                            "table cell is %d words, cap is %d (split the "
                            "cell into bullets or a diagram)"
                            % (wc, TABLE_CELL_MAX_WORDS)))
            # Column alignment. Flag rows whose pipe separators do not line
            # up with the shared column widths.
            #
            # - Only padded rows are candidates, so an unpadded row stays
            #   table-mispadding's job and never double-fires here.
            # - A run with ragged cell counts is table-mispadding's job too,
            #   so it is skipped entirely.
            candidates = []
            for row in run:
                if not _table_row_padded(row[1]):
                    continue
                _prefix, cells = _split_table_row(row[1])
                if cells:
                    candidates.append((row, cells))
            if candidates:
                n_cols = len(candidates[0][1])
                if all(len(cells) == n_cols for _row, cells in candidates):
                    widths = [0] * n_cols
                    for _row, cells in candidates:
                        if _separator_cells(cells):
                            continue
                        for idx, cell in enumerate(cells):
                            if len(cell) > widths[idx]:
                                widths[idx] = len(cell)
                    for row, cells in candidates:
                        if _render_aligned_row("", cells, widths) != row[1]:
                            findings.append(Finding(
                                path, row[0], "table-alignment",
                                "table columns are not aligned (pad each "
                                "cell to the column width so the | "
                                "separators line up)"))
        i = j


# Flow and step vocabulary the missing-diagram advisory matches.
DIAGRAM_FLOW_VOCAB = re.compile(
    r"\b(?:step|transition|compile|walk|pipeline|stage|phase|flow|lifecycle|"
    r"then|sequence|match)\b", re.IGNORECASE)


def _bullet_lists_in_block(block):
    """Counts the maximal bullet lists in a block, the runs `check_bullets`
    treats as one list each."""
    lead_idx = {}
    for i, (_n, c, kind, indent, trailing) in enumerate(block):
        if kind in ("doc", "hash", "prose") and c and not trailing \
                and BULLET_RE.match(c):
            lead_idx[i] = indent
    if not lead_idx:
        return 0
    count = 0
    i = 0
    while i < len(block):
        if i not in lead_idx:
            i += 1
            continue
        count += 1
        j = i
        while j < len(block):
            if j in lead_idx:
                k = j + 1
                while k < len(block) and k not in lead_idx \
                        and _bullet_continuation(block[k], lead_idx[j]):
                    k += 1
                j = k
                continue
            break
        i = j + 1
    return count


def check_missing_diagram(path, block, findings):
    """Fires the advisory missing-diagram rule on a comment block that
    describes a multi-step flow in bullets and prose with no diagram.

    A block qualifies when it holds 2 or more bullet lists, carries flow
    or step vocabulary, and contains no diagram line. The finding is
    advisory, so it prints without blocking a commit.
    """
    kinds = {e[2] for e in block}
    if not (kinds & {"doc", "hash"}):
        return
    if any(is_diagram(c) for _n, c, _k, _i, _t in block if c):
        return
    text = " ".join(c for _n, c, _k, _i, _t in block if c)
    if not DIAGRAM_FLOW_VOCAB.search(text):
        return
    if _bullet_lists_in_block(block) < 2:
        return
    findings.append(Finding(
        path, block[0][0], "missing-diagram",
        "multi-step flow described in prose/bullets; "
        "consider a sequence or dataflow diagram", warning=True))


def check_module_header(path, header_lines, is_test, is_self, findings):
    """Runs the module header rules, the tight-line cap and the test run command.

    Contract:
    - the linter's own header stays exempt from the tight-line cap
    - it carries the rule table and the golden rules required of every
      linter file, the same exemption the skill's own rule definitions hold
    """
    if not header_lines:
        return
    first_line = header_lines[0][0]
    prose = [(n, c) for n, c in header_lines if c and not structural(c)]
    if len(prose) > MODULE_HEADER_MAX_LINES and not is_self:
        findings.append(Finding(
            path, first_line, "module-header-length",
            "module header runs %d tight prose lines, cap is %d "
            "(bullets, tables, and diagrams may extend beyond)"
            % (len(prose), MODULE_HEADER_MAX_LINES), warning=True))
    if is_test and not any(RUN_CMD_RE.search(c) for _, c in header_lines):
        findings.append(Finding(
            path, first_line, "test-header-command",
            "test file header carries no run command line"))


def _type_decl_and_field_lines(lines):
    """Classifies a Nim file's type-section lines.

    Returns the (decls, fields) pair, two sets of 1-based line numbers.

    - decls, the type-section member declarations, the first member's
      indentation is the member level and deeper lines are bodies
    - fields, the object-body lines indented deeper than their member

    A single-line `type X = object` declaration counts as a decl by itself.
    Blank and comment lines are neutral, they end neither the section nor
    a member body.
    """
    decls = set()
    fields = set()
    n = len(lines)
    i = 0
    while i < n:
        if not TYPE_KEYWORD_RE.match(lines[i]):
            i += 1
            continue
        if "=" in lines[i]:
            decls.add(i + 1)
            i += 1
            continue
        j = i + 1
        member_indent = None
        while j < n:
            stripped = lines[j].strip()
            if not stripped or stripped.startswith("#"):
                j += 1
                continue
            indent = len(lines[j]) - len(lines[j].lstrip())
            if member_indent is None:
                member_indent = indent
            elif indent < member_indent:
                break
            if indent == member_indent:
                decls.add(j + 1)
            else:
                fields.add(j + 1)
            j += 1
        i = j
    return decls, fields


def _type_has_field_docs(lines, decl_idx):
    """Returns True when the type body carries ## doc lines, the field-doc
    placement the doc-above-type rule fixes toward.

    The body spans the lines indented deeper than the declaration, up to
    the first line at or below the declaration's indent."""
    decl_indent = len(lines[decl_idx]) - len(lines[decl_idx].lstrip())
    j = decl_idx + 1
    n = len(lines)
    while j < n:
        stripped = lines[j].strip()
        if not stripped:
            j += 1
            continue
        indent = len(lines[j]) - len(lines[j].lstrip())
        if indent <= decl_indent:
            break
        if stripped.startswith("##"):
            return True
        j += 1
    return False


def check_doc_above_type(path, text, findings):
    """Fires the doc-above-type rule on a ## block placed directly above a type declaration.

    Flagged when the line immediately after the ## block is one of:

    - the `type` keyword, section header or single-line declaration
    - a type-section member declaration (object, ref object, distinct, enum, tuple)
    - the bare `X* = object` single-line form outside a section
    - a module doc block directly above a leading type declaration, Nimdoc
      attaches it to the type, never the intended attachment

    Not flagged:

    - ## field docs inside a type body
    - ## above procs, funcs, consts, lets, and vars
    - a ## block with any line between it and the declaration, a blank line
      is the stray-comment case and a # comment line is the convert-to-# fix
    """
    lines = text.splitlines()
    decls, fields = _type_decl_and_field_lines(lines)
    n = len(lines)
    i = 0
    while i < n:
        if not lines[i].strip().startswith("##"):
            i += 1
            continue
        start = i
        while i < n and lines[i].strip().startswith("##"):
            i += 1
        if i >= n:
            continue
        code = lines[i]
        k = i
        if code.strip() == "":
            # a blank line breaks the adjacency, the stray-comment case
            continue
        if (TYPE_KEYWORD_RE.match(code)
                or (k + 1) in decls
                or (TYPE_DECL_RE.match(code) and (k + 1) not in fields)):
            findings.append(Finding(
                path, start + 1, "doc-above-type",
                "## above the type declaration: the doc comment belongs "
                "inside the body, above the fields it describes"))


SIG_HEAD_RE = re.compile(r"^\s*(?:proc|func)\b")
SIG_WRAP_MAX = 140


def nim_sig_wrap_checks(path, text, findings):
    """Flags a proc or func signature wrapped across lines while the joined
    single-line form fits the 100-column code budget. Only the comma-join
    shape counts, a first line ending on a comma and continuations that
    complete the parameter list; a signature too long to join stays legal."""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        if not SIG_HEAD_RE.match(raw) or not raw.rstrip().endswith(","):
            continue
        parts = [raw.strip()]
        j = i + 1
        while j < len(lines) and j <= i + 8:
            seg = lines[j].strip()
            if not seg:
                break
            parts.append(seg)
            if not seg.endswith(","):
                break
            j += 1
        if len(parts) < 2 or parts[-1].endswith(","):
            continue
        joined = " ".join(p for seg in parts for p in seg.split())
        if len(joined) <= SIG_WRAP_MAX:
            findings.append(Finding(
                path, i + 1, "sig-wrap",
                "the signature fits one %d-char line (%d joined), do not wrap"
                % (SIG_WRAP_MAX, len(joined))))


# The file where a try block is the sanctioned exception boundary: the libtorch
# FFI translation seam. Tests folders are exempt wholesale (test harnesses,
# fuzz loops, C++ capture).
TRY_ALLOWLIST = (
    "workspace/libtorch/src/tensors.nim",
)
TRY_RE = re.compile(r"^\s*try\s*:$")

# Justification markers: a doc comment exists to state the contract a caller
# must know, never to walk a reviewer through why the code reads as it does.
DESIGN_NARRATION_RE = re.compile(
    r"\b(?:because|instead of|rather than|which is why)\b"
    r"|\b(?:declared|defined)\s+(?:ahead|before|after|above|below)\b"
    r"|\band not\b|\bbut not\b", re.IGNORECASE)


def nim_design_narration_checks(path, text, findings):
    """Flags doc-comment and whole-line maintainer-comment lines that justify
    the design to the audience - the because-clause, the X-and-not-Y contrast,
    the layout-position story (declared ahead of the steps, defined above) -
    instead of stating the contract. The caller's test for doc content: what
    can they do with it? A layout decision answers nothing (the LSP shows the
    declaration), and the why of a choice dies with the choice; only
    caller-visible constraints survive in the contract."""
    for i, raw in enumerate(text.splitlines()):
        m = re.match(r"^\s*##(.*)$", raw) or re.match(r"^\s*#(?!#)(.*)$", raw)
        if m and DESIGN_NARRATION_RE.search(m.group(1)):
            findings.append(Finding(
                path, i + 1, "design-narration",
                "the doc justifies the design (because, instead of, X and "
                "not Y) instead of stating the contract, state what the "
                "caller must know"))


HASH_SEPARATOR_RE = re.compile(r"^\s*#\s*-+\s*$")


def nim_section_separator_checks(path, text, findings):
    """Flags whole-line `#` comments built from dashes. A dash rule is a
    layout-position marker: it says where a section sits on the screen, not
    anything a reader of the line needs, and it dies when the code moves.
    The section title line above or below carries the same information."""
    for i, raw in enumerate(text.splitlines()):
        if HASH_SEPARATOR_RE.match(raw):
            findings.append(Finding(
                path, i + 1, "section-separator",
                "a dash rule is a layout-position marker (keep the section "
                "title, drop the rule)"))


def try_allowed(path):
    norm = str(path).replace("\\", "/")
    if norm.endswith(tuple(TRY_ALLOWLIST)) or norm in TRY_ALLOWLIST:
        return True
    parts = norm.split("/")
    return "tests" in parts


def nim_try_block_checks(path, text, findings):
    """Flags every try/except or try/finally block outside the sanctioned
    boundaries and tests folders. Raising exceptions is fine anywhere; the
    ban is on catching them as control flow, which turns error paths into
    hidden gotos - worst of all catching and re-raising the same exception
    (zstd) or swallowing it (chattyninja filter). Only C++ FFI translation
    (libtorch tensors.nim) legitimately catches."""
    if try_allowed(path):
        return
    for i, raw in enumerate(text.splitlines()):
        if TRY_RE.match(raw):
            findings.append(Finding(
                path, i + 1, "try-block",
                "raising is fine, catching is not: no try/except or "
                "try/finally as control flow outside the C++ boundary "
                "(libtorch tensors.nim) and tests folders"))


EXCEPT_RE = re.compile(r"^\s*except\b([^:]*):")
EXCEPT_AS_RE = re.compile(r"\bas\s+(\w+)\s*$")
RAISE_NAMED_RE = re.compile(r"^\s*raise\s+(\w+)\s*$")
RAISE_BARE_RE = re.compile(r"^\s*raise\s*$")


def nim_except_rewrap_checks(path, text, findings):
    """Flags an except clause whose body re-raises the caught exception, the
    rewrap shape that pays try/except cost in the hot path to rewrite a
    message. Raising a different exception is translation and stays legal."""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        m = EXCEPT_RE.match(raw)
        if not m:
            continue
        name_m = EXCEPT_AS_RE.search(m.group(1))
        name = name_m.group(1) if name_m else None
        indent = len(raw) - len(raw.lstrip())
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if not s.strip():
                j += 1
                continue
            if len(s) - len(s.lstrip()) <= indent:
                break
            bare = RAISE_BARE_RE.match(s)
            named = RAISE_NAMED_RE.match(s)
            if bare or (name and named and named.group(1) == name):
                findings.append(Finding(
                    path, j + 1, "except-rewrap",
                    "the except clause re-raises the caught exception, handle "
                    "it or let it propagate, never rewrap in the hot path"))
                break
            j += 1


def nim_structure_checks(path, text, header_nos, findings):
    """Runs the Nim structure rules over the declarations, the shapes
    stay conservative single-line forms.
    - doc-above-proc bans the ## block above a proc or func declaration
    - the house doc comment of a proc or func is the first body line
    - missing-doc plus missing-contract read the body doc block
    - exported types are documented by a doc block above them or by the
      ## field docs inside the body, a ## block directly above the
      declaration is banned (doc-above-type)"""
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        m = NIM_EXPORTED_CALLABLE_RE.match(line)
        above_banned = i > 0 and lines[i - 1].strip().startswith("##") \
            and i not in header_nos
        if m:
            if above_banned:
                findings.append(Finding(
                    path, i + 1, "doc-above-proc",
                    "## above the declaration: the house doc comment is "
                    "the first body line"))
            # the declaration may span several lines, the body opens after
            # the signature terminates on its '=' or ':' line, mid-signature
            # defaults such as `depth = 1,` are no terminators
            body = None
            if line[m.end():].rstrip().endswith("="):
                body = i + 1
            else:
                # the '=' terminator wins, a ':' can sit at the end
                # of a signature line whose return type continues on
                # the following line
                j = i + 1
                colon = None
                while j < len(lines) and j <= i + 15:
                    s = lines[j].rstrip()
                    if s.endswith("="):
                        body = j + 1
                        break
                    if s.endswith(":") and colon is None:
                        colon = j + 1
                    j += 1
                if body is None:
                    body = colon
            if body is not None and body < len(lines) \
                    and lines[body].strip().startswith("##"):
                prose = []
                j = body
                while j < len(lines) and lines[j].strip().startswith("##"):
                    c = lines[j].strip()[2:].strip()
                    if c and not structural(c):
                        prose.append(c)
                    j += 1
                if (len(prose) >= 3
                        and not any(CONTRACT_MARKER_RE.search(c)
                                    for c in prose)):
                    findings.append(Finding(
                        path, body + 1, "missing-contract",
                        "multi-line doc states no contract marker "
                        "(add Args, Returns, Precondition, or bullet the contract)"))
            else:
                findings.append(Finding(
                    path, i + 1, "missing-doc",
                    "exported %s carries no doc comment, the doc comment is "
                    "the first body line" % m.group(1)))
            continue
        if re.match(r"^\s*(?:proc|func)\b", line):
            if above_banned:
                findings.append(Finding(
                    path, i + 1, "doc-above-proc",
                    "## above the declaration: the house doc comment is "
                    "the first body line"))
            continue
        tm = NIM_EXPORTED_TYPE_RE.match(line)
        if not tm:
            continue
        doc_lines = []
        j = i - 1
        depth = 0
        while j >= 0 and depth < 5:
            if (j + 1) in header_nos:
                # The module header block is consumed as the module header,
                # it never documents the declaration below it.
                break
            prev = lines[j].strip()
            if prev.startswith("##") or prev.startswith("///"):
                doc_lines.append((j + 1, prev.lstrip("#/").strip()))
            elif prev:
                break
            j -= 1
            depth += 1
        if not doc_lines and not _type_has_field_docs(lines, i):
            findings.append(Finding(
                path, i + 1, "missing-doc",
                "exported %s carries no doc comment" % tm.group(1)))
            continue
        prose = [(n, c) for n, c in reversed(doc_lines) if c and not structural(c)]
        if (len(prose) >= 3
                and not any(CONTRACT_MARKER_RE.search(c) for _, c in prose)):
            findings.append(Finding(
                path, prose[0][0], "missing-contract",
                "multi-line doc states no contract marker "
                "(add Args, Returns, Precondition, or bullet the contract)"))


def py_structure_checks(path, tree, func_docs, findings):
    """Runs the Python structure rules.
    Missing-doc covers module-level public defs and classes.
    Missing-contract covers multi-line function docs."""
    if tree is None:
        return
    documented = set()
    for node in tree.body:
        if (isinstance(node, _PY_DOCSTRING_OWNERS) and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            documented.add(node.body[0].value.lineno)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_") or node.name.startswith("test"):
            continue
        if node.body[0].lineno not in documented:
            findings.append(Finding(
                path, node.lineno, "missing-doc",
                "public %s carries no docstring" % node.name))
    for start, contents in func_docs:
        prose = [c for c in contents if c and not structural(c)]
        if (len(prose) >= 3
                and not any(CONTRACT_MARKER_RE.search(c) for c in contents)
                and not any(BULLET_RE.match(c) for c in contents)):
            findings.append(Finding(
                path, start, "missing-contract",
                "multi-line doc states no contract marker "
                "(add Args, Returns, Precondition, or bullet the contract)"))


def scan(path, text, findings):
    """Runs every rule over one file's prose, appending to findings."""
    entries, meta = extract(path, text)
    is_nim = path.suffix == ".nim"
    is_py = path.suffix == ".py"
    is_md = path.suffix == ".md"
    is_test = bool(NIM_TEST_FILE_RE.search(str(path))
                   or PY_TEST_FILE_RE.search(str(path)))
    is_self = path.resolve() == Path(__file__).resolve()
    blocks = []
    cur = []
    prev_no = None
    for entry in entries:
        if prev_no is not None and entry[0] != prev_no + 1:
            blocks.append(cur)
            cur = []
        cur.append(entry)
        prev_no = entry[0]
    if cur:
        blocks.append(cur)

    if not is_md:
        header_lines = meta["module_header"] if is_py else None
        if header_lines is None:
            for block in blocks:
                kinds = {e[2] for e in block}
                if "doc" in kinds:
                    header_lines = [(e[0], e[1]) for e in block if e[2] == "doc"]
                    break
        check_module_header(path, header_lines or [], is_test, is_self, findings)
        if is_nim:
            header_nos = set()
            for block in blocks:
                if {e[2] for e in block} == {"doc"}:
                    header_nos = {e[0] for e in block}
                    break
            nim_structure_checks(path, text, header_nos, findings)
            nim_sig_wrap_checks(path, text, findings)
            nim_except_rewrap_checks(path, text, findings)
            nim_try_block_checks(path, text, findings)
            nim_design_narration_checks(path, text, findings)
            nim_section_separator_checks(path, text, findings)
            check_doc_above_type(path, text, findings)
        if is_py and meta is not None:
            tree = None
            try:
                tree = ast.parse(text)
            except SyntaxError:
                pass
            py_structure_checks(path, tree, meta["func_docs"], findings)

    prev_prose = None
    prev_text = {}
    for n, c, kind, _, _ in entries:
        prev_text[n] = c
    for block in blocks:
        kinds = {e[2] for e in block}
        block_is_doc = "doc" in kinds
        check_bullets(path, block, findings)
        check_tables(path, block, findings)
        check_missing_diagram(path, block, findings)
        first = next(((e[0], e[1], e[2]) for e in block if e[1] and e[2] != "heading"),
                     None)
        if first and (block_is_doc or first[2] == "hash") \
                and strip_backticks(first[1]).split()[:1] == ["The"]:
            findings.append(Finding(
                path, first[0], "the-opener",
                "comment opens with The (open with a noun phrase or Returns ...)"))
        run = []
        air_run = []
        prev_bullet = False
        for n, c, kind, indent, trailing in block:
            if kind == "heading":
                if strip_backticks(c).split()[:1] == ["The"]:
                    findings.append(Finding(
                        path, n, "the-opener",
                        "heading opens with The (open with a noun phrase)"))
                flush_wall(path, run, findings)
                flush_wall_no_air(path, air_run, findings)
                run, air_run, prev_bullet = [], [], False
                continue
            if not c:
                flush_wall(path, run, findings)
                flush_wall_no_air(path, air_run, findings)
                run, air_run, prev_bullet = [], [], False
                continue
            if kind == "fence":
                # Fenced lines never join prose runs.
        # The structure rules see code layout they cannot judge.
                flush_wall(path, run, findings)
                flush_wall_no_air(path, air_run, findings)
                run, air_run, prev_bullet = [], [], False
                check_line(path, n, c, kind, is_nim, prev_text, findings)
                continue
            check_line(path, n, c, kind, is_nim, prev_text, findings)
            if structural(c):
                flush_wall(path, run, findings)
                run = []
            if block_is_doc or not is_nim:
                run.append((n, c))
            if trailing:
                # A trailing comment anchors to its code item and is
                # itemized by the code layout, so it ends the run
                # instead of joining it.
                flush_wall_no_air(path, air_run, findings)
                air_run, prev_bullet = [], False
            elif block_is_doc or not is_nim:
                if structural(c):
                    air_run.append((n, True))
                    prev_bullet = bool(BULLET_RE.match(c))
                else:
                    continuation = indent > 0 and prev_bullet
                    air_run.append((n, continuation))
                    prev_bullet = continuation
            if structural(c):
                continue
            wc = len(words(strip_backticks(c)))
            if (wc <= 2 and not c.endswith(":") and prev_prose is not None
                    and len(prev_prose[1]) < SINGLE_WORD_EOL_PREV_MAX):
                findings.append(Finding(
                    path, n, "single-word-eol",
                    "line ends on a %d-word stub, the previous line had room "
                    "to reflow" % wc))
            prev_prose = (n, c)
        flush_wall(path, run, findings)
        flush_wall_no_air(path, air_run, findings)


def _repo_root():
    """Returns the repository root path, or None outside a git repository."""
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    root = out.stdout.strip()
    return root if out.returncode == 0 and root else None


def _parse_added_lines(diff):
    """Extracts the 1-based post-image line numbers a unified=0 diff adds."""
    lines = set()
    new_ln = None
    for raw in diff.split("\n"):
        if raw.startswith("@@"):
            plus = raw.split("+", 1)[1].split("@@", 1)[0]
            new_ln = int(plus.split(",")[0])
        elif raw.startswith("+") and not raw.startswith("+++"):
            if new_ln is not None:
                lines.add(new_ln)
                new_ln += 1
        elif raw.startswith("\\"):
            continue
        elif new_ln is not None:
            # context lines advance the post-image cursor, removals do not
            if raw.startswith(" "):
                new_ln += 1
    return lines


def _diff_added_map(base):
    """Returns {post-image path: set of added post-image line numbers}.

    The whole diff is read in one pass and bucketed per file, because a
    pathspec limited to one name prevents git from pairing a rename, which
    would report an unchanged renamed file as entirely new. `--find-renames`
    makes a pure rename contribute no added lines, and a rename with edits
    contributes only its hunks.
    """
    cmd = ["git", "diff", "--find-renames", "--unified=0"]
    if base:
        cmd.append(base)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    added = {}
    for chunk in out.stdout.split("diff --git "):
        if not chunk.strip():
            continue
        path = None
        for line in chunk.split("\n"):
            if line.startswith("+++ b/"):
                path = line[6:].split("\t", 1)[0]
                break
        if path is None:
            continue
        added[path] = added.get(path, set()) | _parse_added_lines("diff --git " + chunk)
    return added


def added_lines(path, base, cache=None):
    """Returns the 1-based post-image line numbers a diff from base adds.

    The diff is read from the working tree relative to the base commit
    (`git diff <base> --unified=0`), the same text the linter reads, so a
    partially-staged file keeps its line numbers aligned. Returns None
    when there is no repository to read a diff from, or when git fails, so
    the caller scopes nothing and lints the whole file instead of silently
    passing on an empty added-line set.

    `cache` is a per-invocation dict, one fresh dict per lint call, so a
    whole lint still reads the diff once. A caller that omits it gets a diff
    read of its own, never the added-line map of an earlier working-tree
    state.
    """
    repo = _repo_root()
    if repo is None:
        return None
    rel = os.path.relpath(path.resolve(), repo)
    if rel.startswith(".."):
        return None
    if cache is None:
        cache = {}
    key = (repo, base)
    if key not in cache:
        cache[key] = _diff_added_map(base)
    added = cache[key]
    if added is None:
        return None
    return added.get(rel, set())


def collect_files(paths):
    """Collects the .nim, .py, and .md files under paths, sorted and deduped.

    Returns every collected file path.
    Excluded are the linter's own file and the writing-docs skill's
    rule definitions, which stay exempt from their own checks.
    """
    files = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            sys.stderr.write("lint_docs: missing path: %s\n" % p)
            sys.exit(2)
        if pp.is_dir():
            for ext in (".nim", ".py", ".md"):
                files.extend(sorted(pp.rglob("*" + ext)))
        elif pp.suffix in (".nim", ".py", ".md"):
            files.append(pp)
    this_file = Path(__file__).resolve()
    # EXAMPLES.md stays the one exempt file since it cites the banned
    # forms as teaching material and would otherwise flag itself.
    #
    # SKILL.md and REFERENCE.md comply with their own rules like
    # every other file.
    examples = (this_file.parent.parent / "EXAMPLES.md").resolve()
    out = []
    seen = set()
    for f in sorted(set(files)):
        rf = f.resolve()
        if rf == this_file or rf == examples or rf in seen:
            continue
        seen.add(rf)
        out.append(f)
    return out


def lint(paths, base=None):
    """Lints every collected file under paths, returning sorted findings.

    When base is set, findings are scoped to the added lines of the diff
    from that commit to the staged index (the working tree when nothing
    is staged), so pre-existing violations a change did not touch stay
    out of the report. Without base, every file is linted in full.
    """
    findings = []
    added_lines_cache = {}
    for f in collect_files(paths):
        if not f.is_file():
            # A path deleted by the change under review has no lines to scope to.
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        file_findings = []
        scan(f, text, file_findings)
        if base:
            added = added_lines(f, base, added_lines_cache)
            if added is None:
                findings.extend(file_findings)
            else:
                findings.extend(fd for fd in file_findings
                                if fd.line in added)
        else:
            findings.extend(file_findings)
    findings.sort(key=lambda x: (str(x.path), x.line, x.rule))
    return findings


def lint_text(text, filename):
    """Lints one in-memory source text, the library entry point
    the sibling linters import. Returns a sorted list of Finding records."""
    findings = []
    scan(Path(filename), text, findings)
    findings.sort(key=lambda x: (str(x.path), x.line, x.rule))
    return findings


# The counted rule classes the autofix rewrites, comment text only.
# Every other class needs judgment about what the prose should say.
MECHANICAL_RULES = frozenset((
    "article-eol", "stray-fragment", "single-word-eol",
    "unit-split", "colon-inline", "table-alignment",
))

# wall-no-air joins the transform triggers for the one-sentence merge but
# stays a judgment class for the exit verdict.
_MERGE_RULE = "wall-no-air"

# The wrap width never drops under this floor, so a file of short prose
# lines still reflows into readable paragraphs.
_FIX_WIDTH_FLOOR = 40

# The continuation indent a colon split puts under its lead line.
_FIX_INDENT_STEP = 2

# The accepted-transform budget of one fix_file call sits far above
# the mechanical site count of any real file.
_FIX_ROUND_CAP = 1000

# Wrap tokens split on whitespace with one backticked span kept whole
# as one token, a None entry marks a hard break the rewrap keeps.
_SPAN_TOKEN_RE = re.compile(r"`[^`]*`[.,;:()'\"*]?|\S+")

# A noqa comment lead, tooling metadata the prose rules skip
NOQA_RE = re.compile(r"\s*noqa\b")

# A line opening on a severed `word,` continuation (the unit-split shape).
_COMMA_LEAD_RE = re.compile(r"^[a-z]+,")

# A lowercase single-word colon lead, the colon-break shape the house
# rules keep with their lead phrase.
_COLON_TOKEN_RE = re.compile(r"^\w{1,15}:")


def _fix_width(entries):
    """Returns the wrap width of one file, the longest plain prose line
    capped at the prose cap, floored so short files still read well."""
    lens = [len(c) for n, c, kind, ind, tr in entries
            if c and kind in ("doc", "hash", "prose")
            and not structural(c) and not code_like(c)]
    cap = max(lens) if lens else 0
    return max(_FIX_WIDTH_FLOOR, min(cap, PROSE_CAP))


def _prose_tokens(text):
    """Splits prose into wrap tokens, one backticked span per token."""
    return _SPAN_TOKEN_RE.findall(text)


def _tail_bad(body):
    """Returns True when a wrapped line would trip one of the counted
    prose rules at its end.
    - a dangling article, before a code span too
    - a stranded possessive or a bare connective
    - a 1-2 word tail after a sentence period"""
    bare = strip_backticks(body)
    endtxt = re.sub(r"`[^`]*`", " XcodeX ", body)
    tok = eol_token(endtxt)
    if tok == "xcodex":
        prev = re.sub(r"XcodeX\s*$", "", endtxt.rstrip()).rstrip()
        if eol_token(prev) in ARTICLE_EOL:
            return True
    elif tok:
        if tok in ARTICLE_EOL or tok.endswith("'s") or tok in CONNECTIVE_EOL:
            return True
    m = re.search(r"(?<![.\d])\.\s+(\S+(?:\s+\S+)?)$", bare)
    if m:
        lead = m.group(1).split()[0].strip(".,;:()'\"*").lower()
        if lead not in ABBREVIATIONS and "`" not in m.group(1):
            return True
    return False


def _wrap_tokens(tokens, width, indent):
    """Rewraps prose tokens into lines of at most width columns.

    Expected input:
    - tokens, the whitespace-separated words with each backticked span
      kept whole as one token, a None entry marks a hard break where
      the author ended a line on a colon
    - width, the column budget every line wraps under
    - indent, the wrap indent every line carries, counted against width

    Output:
    - the wrapped body lines without the indent
    - None when no clean wrap exists, from a lone token over the cap,
      an unhealable stub, or a break that would orphan a lowercase
      `label:` lead away from a sentence terminator

    Break discipline, mirroring the counted prose rules:
    - a line never ends on a dangling article or a stranded possessive
    - a line never ends on a bare connective or a sentence-period tail
      of 1-2 words
    - a closed line keeps 3 or more words, so no stub line forms
    - a severed `word,` unit never opens a line, the comma token joins
      the line above it instead
    - the final stub heals by pulling words down from the line above it,
      a pull may pass width but stays under the prose cap
    """
    pad = " " * indent
    lines, seg, carry = [], [], []
    i, n = 0, len(tokens)
    while True:
        if carry and not seg:
            seg, carry = carry, []
        if i < n and tokens[i] is None:
            if not seg:
                return None
            lines.append(seg)
            seg, carry = [], []
            i += 1
            continue
        if i >= n:
            break
        while (i < n and tokens[i] is not None
               and len(pad + " ".join(seg + [tokens[i]])) <= width):
            seg.append(tokens[i])
            i += 1
        if i < n and tokens[i] is None:
            lines.append(seg)
            seg, carry = [], []
            i += 1
            continue
        if i >= n:
            break
        # A break is needed before tokens[i], the line tail heals first.
        while seg and _tail_bad(pad + " ".join(seg)):
            carry.insert(0, seg.pop())
        if not seg:
            return None
        if _COMMA_LEAD_RE.match(strip_backticks(tokens[i])):
            # The severed comma unit never opens a line, the wrap
            # closes above it or joins it up when the join fits width.
            if len(seg) > 3 and not _tail_bad(pad + " ".join(seg[:-1])):
                carry.insert(0, seg.pop())
                lines.append(seg)
                seg = []
                continue
            seg.extend(carry)
            carry = []
            seg.append(tokens[i])
            i += 1
            if (_tail_bad(pad + " ".join(seg))
                    or len(pad + " ".join(seg)) > width):
                return None
            continue
        nxt_bare = strip_backticks(carry[0] if carry else tokens[i])
        if (_COLON_TOKEN_RE.match(nxt_bare) and i + 1 < n
                and not re.search(r"[.:;]\s*$", pad + " ".join(seg))):
            # Breaking orphans the colon lead while joining keeps it
            # inline on one line, so no mechanical remedy exists here.
            return None
        if len(seg) < 3:
            seg.extend(carry)
            carry = []
            seg.append(tokens[i])
            i += 1
            if len(pad + " ".join(seg)) > PROSE_CAP:
                return None
            continue
        lines.append(seg)
        seg = []
    if seg or carry:
        lines.append(seg if seg else carry)
    if len(lines) >= 2 and not lines[-1][-1].endswith(":"):
        for _ in range(8):
            last, prev = lines[-1], lines[-2]
            if len(last) > 2 and not _tail_bad(pad + " ".join(prev)):
                break
            if len(prev) <= 3 or prev[-1].endswith(":"):
                break
            last.insert(0, prev.pop())
        if len(lines[-1]) <= 2 or len(pad + " ".join(lines[-1])) > PROSE_CAP:
            return None
        if (_tail_bad(pad + " ".join(lines[-1]))
                or _tail_bad(pad + " ".join(lines[-2]))):
            return None
    return [" ".join(seg) for seg in lines]


class _ProseLine:
    """One plain prose comment line the autofix may rewrite.

    - no is the 1-based file line number
    - text is the comment body with the wrap indent stripped
    - indent is the wrap indent width the rewrap preserves
    - emit is the raw prefix (indent excluded) every re-emitted line carries
    - head is the raw prefix of this exact line when it anchors a frame,
      a docstring opening quote, None for plain lines
    - tail is the raw suffix of this exact line when it anchors a frame,
      a docstring closing quote, None for plain lines
    - ends_backslash marks a raw line the next line cannot join
    """
    __slots__ = ("no", "text", "indent", "emit", "head", "tail",
                 "ends_backslash", "kind")

    def __init__(self, no, text, indent, emit, head, tail, ends_backslash,
                 kind):
        self.no, self.text, self.indent = no, text, indent
        self.emit, self.head, self.tail = emit, head, tail
        self.ends_backslash, self.kind = ends_backslash, kind


def _py_line_frames(lines, meta):
    """Maps each docstring body line to its raw frames for re-emission.

    Returns a dict of line number to (emit, head, tail) triples where emit holds the plain base
    indent prefix, head and tail hold the opening and closing quote frames
    on the outer body lines.

    A docstring whose frames fail reconstruction maps to None for every
    body line, the autofix then skips those lines entirely.
    """
    out = {}
    for start, end, base, quote, _is_module in meta["docstring_spans"]:
        if not quote:
            for n in range(start, end + 1):
                out[n] = None
            continue
        first_raw = lines[start - 1]
        m = _PY_OPEN_QUOTE_RE.match(first_raw[base:])
        head = None
        if m and m.group(2) == quote:
            head = first_raw[:base] + m.group(1) + quote
        tail = None
        if end != start:
            last_raw = lines[end - 1].rstrip()
            lead = len(last_raw) - len(last_raw.lstrip())
            sl = last_raw[min(lead, base):]
            tail = sl[len(_strip_string_frame(sl, quote)):]
        else:
            sl = first_raw.rstrip()[base:]
            sl = re.sub(r"^[A-Za-z]*", "", sl, count=1)
            if sl.startswith(quote):
                body_sl = sl[len(quote):]
                tail = body_sl[len(_strip_string_frame(body_sl, quote)):]
        for n in range(start, end + 1):
            out[n] = (None if head is None else
                      (" " * base, head if n == start else None,
                       tail if n == end else None))
    return out


def _nim_record(n, c, kind, indent, raw):
    """Builds one whole-line Nim comment record, None for a raw shape the emit cannot reconstruct."""
    s = raw.strip()
    for marker in ("##", "///", "#"):
        if s.startswith(marker):
            if c.startswith(("#", "/")):
                return None
            ws = raw[:len(raw) - len(raw.lstrip())]
            return _ProseLine(n, c, indent, ws + marker, None, None,
                              raw.rstrip().endswith("\\"), kind)
    return None


def _py_hash_record(n, c, kind, indent, raw, hash_toks):
    """Builds one whole-line Python comment record, None for a raw shape the emit cannot reconstruct."""
    tok = hash_toks.get(n)
    if tok is None:
        return None
    col, s = tok
    k = len(s) - len(s.lstrip("#"))
    return _ProseLine(n, c, indent, raw[:col] + "#" * k, None, None,
                      raw.rstrip().endswith("\\"), kind)


def _fix_paragraphs(path, text):
    """Collects the plain prose paragraphs of one file for the autofix.

    Returns (paragraphs, bullets). Each paragraph is a list of _ProseLine records holding adjacent
    comment lines of one kind at one indent under one raw prefix shape, bullets the lead lines
    the colon split may cut.

    Outside the paragraphs sit the shapes a rewrap could garble:
    - bullets, tables, and diagrams
    - code samples and command blocks
    - trailing comments and unframeable docstring lines
    """
    entries, meta = extract(path, text)
    lines = text.splitlines()
    frames = {}
    hash_toks = {}
    if path.suffix == ".py" and meta:
        frames = _py_line_frames(lines, meta)
        for n, col, s in meta["comment_toks"]:
            hash_toks[n] = (col, s)
    paras, bullets, cur = [], {}, []
    prev_no = None
    for n, c, kind, indent, trailing in entries:
        contiguous = prev_no is not None and n == prev_no + 1
        rec = None
        if (c and kind in ("doc", "hash", "prose") and not trailing
                and not structural(c) and not code_like(c)
                and "->" not in c and not c.startswith(("./", "$"))):
            raw = lines[n - 1]
            if not raw.rstrip().endswith("\\"):
                if path.suffix == ".nim":
                    rec = _nim_record(n, c, kind, indent, raw)
                elif path.suffix == ".py":
                    if kind == "hash":
                        rec = _py_hash_record(n, c, kind, indent, raw,
                                              hash_toks)
                    else:
                        frame = frames.get(n)
                        if frame is not None:
                            emit, head, tail = frame
                            rec = _ProseLine(n, c, indent, emit, head, tail,
                                             raw.rstrip().endswith("\\"),
                                             kind)
                else:
                    ws = raw[:len(raw) - len(raw.lstrip())]
                    rec = _ProseLine(n, c, indent, ws, None, None, False,
                                     kind)
        if rec is not None and cur and (not contiguous
                                        or rec.kind != cur[-1].kind
                                        or rec.indent != cur[-1].indent
                                        or rec.emit != cur[-1].emit
                                        or cur[-1].ends_backslash):
            paras.append(cur)
            cur = []
        if rec is None:
            if cur:
                paras.append(cur)
                cur = []
        else:
            cur.append(rec)
        if rec is None and c and kind in ("doc", "hash", "prose") \
                and not trailing and BULLET_RE.match(c):
            raw = lines[n - 1]
            if not raw.rstrip().endswith("\\"):
                if path.suffix == ".nim":
                    brec = _nim_record(n, c, kind, indent, raw)
                elif path.suffix == ".py":
                    if kind == "hash":
                        brec = _py_hash_record(n, c, kind, indent, raw,
                                               hash_toks)
                    else:
                        frame = frames.get(n)
                        if frame is not None:
                            emit, head, tail = frame
                            brec = _ProseLine(n, c, indent, emit, head,
                                              tail,
                                              raw.rstrip().endswith("\\"),
                                              kind)
                        else:
                            brec = None
                else:
                    ws = raw[:len(raw) - len(raw.lstrip())]
                    brec = _ProseLine(n, c, indent, ws, None, None, False,
                                      kind)
                if brec is not None:
                    bullets[n] = brec
        prev_no = n
    if cur:
        paras.append(cur)
    return paras, bullets


def _emit_paragraph(para, new_texts):
    """Renders the rewritten body lines with the original frames.

    Expected input:
    - para, the _ProseLine records the new lines replace
    - new_texts, the rewritten body lines without indent

    Output:
    - the raw file lines
    - the first emitted line carries the head frame when the paragraph
      anchors one, the last emitted line carries the tail frame
    - every line takes the emit prefix and indent of the record it
      replaces or, when the line count changed, of the nearest record
      at the same end of the paragraph
    """
    old, out = para, []
    n_old, n_new = len(old), len(new_texts)
    for i, t in enumerate(new_texts):
        if n_new == n_old:
            j = i
        elif n_new > n_old:
            j = min(i, n_old - 1)
        else:
            j = i if i < n_new - 1 else n_old - 1
        rec = old[j]
        emit = old[0].head if (i == 0 and old[0].head is not None) else rec.emit
        tail = old[-1].tail if (i == n_new - 1 and old[-1].tail is not None) else ""
        out.append(emit + " " * rec.indent + t + tail)
    return out


def _single_sentence(joined):
    """Returns True when the joined paragraph text is one sentence."""
    s = re.sub(r"[\"\')]+$", "", joined.strip())
    if not s or s[-1] not in ".!?":
        return False
    bare = re.sub(r"[0-9]\.[0-9]", " ", strip_backticks(s[:-1]))
    return "." not in bare


def _para_candidates(para, width, findings, wall_lines):
    """Returns the candidate body-line lists for one paragraph.

    Expected input:
    - para, the _ProseLine records of one plain prose paragraph
    - width, the wrap budget of the paragraph's lines
    - findings, the file's current findings
    - wall_lines, the line numbers the wall-no-air findings anchor

    A paragraph rewraps when a mechanical finding anchors inside it:
    - the colon split runs first, the rewrap second
    - a wall-no-air block joins only when one sentence rewraps into
      3 or fewer lines, every other wall shape needs judgment
    """
    span = (para[0].no, para[-1].no)
    mech = [f for f in findings
            if f.rule in MECHANICAL_RULES and span[0] <= f.line <= span[1]]
    cands = []
    colon_lines = [f.line for f in mech if f.rule == "colon-inline"]
    if colon_lines:
        cand = _colon_split_texts(para, width, colon_lines[0])
        if cand is not None:
            cands.append(cand)
    rewrap_rules = MECHANICAL_RULES - {"colon-inline"}
    if any(f.rule in rewrap_rules for f in mech):
        cand = _rewrap_texts(para, width)
        if cand is not None:
            cands.append(cand)
    if (not cands and any(ln in span for ln in wall_lines)
            and _single_sentence(" ".join(r.text for r in para))):
        cand = _rewrap_texts(para, width)
        if cand is not None and len(cand) <= 3:
            cands.append(cand)
    return cands


def _colon_split_texts(para, width, line_no):
    """Builds the colon-split candidate for one colon-inline line.

    Output:
    - the paragraph's new body lines with the lead ending on its colon
      and the tail rewrapped two spaces deeper
    - None when the tail would form a stub line or the lead itself
      overflows the width
    """
    rec = next((r for r in para if r.no == line_no), None)
    if rec is None:
        return None
    m = _first_prose_colon(rec.text)
    if m is None:
        return None
    head = rec.text[:m.start()].rstrip() + ":"
    tail = rec.text[m.start() + 1:].strip()
    if len(head) < 3 or len(head) > width or len(_prose_tokens(tail)) < 3:
        return None
    tail_lines = _wrap_tokens(_prose_tokens(tail),
                              width - _FIX_INDENT_STEP, 0)
    if tail_lines is None:
        return None
    pad = " " * _FIX_INDENT_STEP
    k = para.index(rec)
    return ([r.text for r in para[:k]] + [head]
            + [pad + ln for ln in tail_lines]
            + [r.text for r in para[k + 1:]])


def _rewrap_texts(para, width):
    """Builds the rewrap candidate for one paragraph.

    Output:
    - the paragraph's new body lines, one per wrapped line
    - None when no clean wrap exists
    Lines the author ended on a colon keep their own line (hard break),
    everything else reflows greedily.
    """
    tokens = []
    for k, rec in enumerate(para):
        tokens.extend(_prose_tokens(rec.text))
        if rec.text.endswith(":") and k < len(para) - 1:
            tokens.append(None)
    width = width - (para[0].indent if para[0].head is not None
                     or para[-1].tail is not None else 0)
    return _wrap_tokens(tokens, width, para[0].indent)


def _first_prose_colon(text):
    """Returns the first prose colon followed by prose, or None when every
    colon sits inside a backtick span."""
    for m in re.finditer(r":\s+\S", text):
        if text[:m.start()].count("`") % 2 == 0:
            return m
    return None


def _bullet_split_texts(rec, width):
    """Builds the colon-split candidate for one bullet lead line, the lead ending on its colon,
    the tail its continuation, None on the usual split guards."""
    m = _first_prose_colon(rec.text)
    if m is None:
        return None
    head = rec.text[:m.start()].rstrip() + ":"
    tail = rec.text[m.start() + 1:].strip()
    if len(head) < 3 or len(head) > width or len(_prose_tokens(tail)) < 3:
        return None
    tail_lines = _wrap_tokens(_prose_tokens(tail),
                              width - _FIX_INDENT_STEP, 0)
    if tail_lines is None:
        return None
    pad = " " * _FIX_INDENT_STEP
    return [head] + [pad + ln for ln in tail_lines]


def _para_width(path, para, width):
    """Returns the wrap budget of one paragraph's lines.

    Docstring lines carry their wrap indent inside the measured content,
    so their budget shrinks by the indent. Whole-line comment markers
    keep the indent out of the measured content.
    """
    if path.suffix == ".py" and para[0].kind == "doc":
        return width - para[0].indent
    return width


def _is_fix_table_line(line, path):
    """Returns True when a raw line is a doc table row the autofix may
    realign. A comment-prefixed table row or a bare row opening on a pipe
    qualifies. A code line that merely carries two pipes never qualifies.
    """
    if not is_table(line):
        return False
    if path.suffix == ".md":
        return True
    stripped = line.strip()
    for marker in ("##", "///", "//", "#"):
        if stripped.startswith(marker):
            rest = stripped[len(marker):].lstrip()
            return rest.startswith("|")
    return stripped.startswith("|")


def _align_table_run(lines, i, j):
    """Aligns one run of table rows to their shared column widths.

    Returns the aligned rendering as a list of lines, or None when the run
    cannot be aligned (ragged cell counts) or is already aligned.
    """
    parsed = []
    n_cols = None
    for k in range(i, j):
        prefix, cells = _split_table_row(lines[k])
        if not cells:
            return None
        if n_cols is None:
            n_cols = len(cells)
        elif len(cells) != n_cols:
            return None
        parsed.append((prefix, cells))
    widths = [0] * n_cols
    for prefix, cells in parsed:
        if _separator_cells(cells):
            continue
        for idx, cell in enumerate(cells):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)
    rendered = []
    for prefix, cells in parsed:
        rendered.append(_render_aligned_row(prefix, cells, widths))
    if rendered == list(lines[i:j]):
        return None
    return rendered


def _fix_first_table_run(path, text):
    """Aligns the first run of unaligned doc table rows in the text.

    Returns (new_text, None) when a run was realigned, (None, None) when no
    qualifying run needs alignment. The fix_file loop applies one table
    run per round.
    """
    raws = text.split("\n")
    n = len(raws)
    i = 0
    while i < n:
        if not _is_fix_table_line(raws[i], path):
            i += 1
            continue
        j = i
        while j < n and _is_fix_table_line(raws[j], path):
            j += 1
        if j - i >= 2:
            aligned = _align_table_run(raws, i, j)
            if aligned is not None:
                return "\n".join(raws[:i] + aligned + raws[j:]), None
        i = j
    return None, None


def _fix_round(path, text, width):
    """Runs one autofix round over one file's text.

    Returns (new_text, 1) when a transform verifies, None when no
    candidate does. A verified transform leaves every rule's finding
    count no higher with the mechanical total strictly dropped.
    """
    findings = []
    scan(path, text, findings)
    mech = [f for f in findings
            if f.rule in MECHANICAL_RULES or f.rule == _MERGE_RULE]
    if not mech:
        return None
    paras, bullets = _fix_paragraphs(path, text)
    before = Counter(f.rule for f in findings)
    wall_lines = {f.line for f in findings if f.rule == _MERGE_RULE}
    raws = text.split("\n")
    for line_no in sorted(f.line for f in findings
                          if f.rule == "colon-inline" and f.line in bullets):
        rec = bullets[line_no]
        texts = _bullet_split_texts(rec, _para_width(path, [rec], width))
        if texts is None:
            continue
        if (sorted(_prose_tokens(rec.text))
                != sorted(t for ln in texts for t in _prose_tokens(ln))):
            continue
        new_text = "\n".join(raws[:line_no - 1]
                             + _emit_paragraph([rec], texts)
                             + raws[line_no:])
        if new_text == text:
            continue
        after = []
        scan(path, new_text, after)
        # Advisory rules never set the exit code, so only the counted
        # rules hold a transform back.
        counts_after = Counter(f.rule for f in after
                               if RULES[f.rule].counted)
        before_counted = Counter(f.rule for f in findings
                                 if RULES[f.rule].counted)
        if any(counts_after[r] > before_counted.get(r, 0)
               for r in counts_after):
            continue
        mech_keys = MECHANICAL_RULES | {_MERGE_RULE}
        if (sum(counts_after[r] for r in mech_keys)
                < sum(before[r] for r in mech_keys)):
            return new_text
    for para in paras:
        span = (para[0].no, para[-1].no)
        for texts in _para_candidates(para, _para_width(path, para, width),
                                      findings, wall_lines):
            if (sorted(t for r in para for t in _prose_tokens(r.text))
                    != sorted(t for ln in texts for t in _prose_tokens(ln))):
                continue
            new_raws = _emit_paragraph(para, texts)
            new_text = "\n".join(raws[:span[0] - 1] + new_raws
                                 + raws[span[1]:])
            if new_text == text:
                continue
            after = []
            scan(path, new_text, after)
            counts_after = Counter(f.rule for f in after)
            if any(counts_after[r] > before.get(r, 0) for r in counts_after):
                continue
            mech_keys = MECHANICAL_RULES | {_MERGE_RULE}
            if (sum(counts_after[r] for r in mech_keys)
                    < sum(before[r] for r in mech_keys)):
                return new_text
    table_fix = _fix_first_table_run(path, text)
    if table_fix[0] is not None:
        after = []
        scan(path, table_fix[0], after)
        counts_after = Counter(f.rule for f in after)
        if any(counts_after[r] > before.get(r, 0) for r in counts_after):
            return None
        mech_keys = MECHANICAL_RULES | {_MERGE_RULE}
        if (sum(counts_after[r] for r in mech_keys)
                < sum(before[r] for r in mech_keys)):
            return table_fix[0]
    return None


def fix_file(path, text):
    """Runs the mechanical autofix over one file's text.

    Returns (new_text, sites), with sites the accepted transform count.
    Each round applies one verified transform, the loop stops on a round
    without a verifying candidate, so a second run changes nothing.
    """
    entries, _meta = extract(path, text)
    width = _fix_width(entries)
    sites = 0
    for _ in range(_FIX_ROUND_CAP):
        nxt = _fix_round(path, text, width)
        if nxt is None:
            break
        text = nxt
        sites += 1
    return text, sites


def fix_one_reported(path):
    """Fixes one file in place and prints the leftover findings tagged
    mechanical or judgment. Returns the mechanical leftover count."""
    text = path.read_text(encoding="utf-8", errors="replace")
    new_text, sites = fix_file(path, text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    remaining = []
    scan(path, new_text, remaining)
    remaining.sort(key=lambda x: (x.line, x.rule))
    for f in remaining:
        if f.rule in MECHANICAL_RULES:
            tag = "mechanical"
        elif f.warning:
            tag = "warning"
        else:
            tag = "judgment"
        print("%s:%d: %s: %s [%s]" % (f.path, f.line, f.rule, f.reason, tag))
    judgment = sum(1 for f in remaining if f.rule not in MECHANICAL_RULES)
    print("%s: fixed %d sites, %d findings remain (LLM-judgment classes)"
          % (path, sites, judgment))
    return sum(1 for f in remaining if f.rule in MECHANICAL_RULES)


_FIX_SELFTEST_SRC = '''def sample():
    """Contract: the tail prose rides on the same line right here.

    One wrapped paragraph line ends on the dangling article the
    next line starts a fresh clause and the reflow heals it.

    A second paragraph drifts along until the wrap leaves a short
    line
    here.

    """
    return 1
'''

_FIX_SELFTEST_FIXED = '''def sample():
    """Contract:
      the tail prose rides on the same line right here.

    One wrapped paragraph line ends on the dangling article
    the next line starts a fresh clause and the reflow heals it.

    A second paragraph drifts along until the wrap leaves
    a short line here.

    """
    return 1
'''

_FIX_SELFTEST_JUDGMENT = '''def judged():
    """The opener prose needs judgment, the mechanical pass leaves
    every docstring line exactly as the author wrote it here."""
    return 3
'''



def main(argv):
    """CLI entry point taking --fix mode, or a file plus directory list.
    Returns the process exit code with 0 for clean, 1 for findings left,
    and 2 for failure."""
    args = argv[1:]
    base = None
    if "--base" in args:
        i = args.index("--base")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        base = args[i + 1]
        args = args[:i] + args[i + 2:]
    base = base or os.environ.get("DOC_LINT_BASE") or None
    if "--fix" in args:
        paths = [a for a in args if a != "--fix"]
        if not paths:
            print(__doc__)
            return 2
        mechanical_left = sum(fix_one_reported(f)
                              for f in collect_files(paths))
        return 1 if mechanical_left else 0
    if not args:
        print(__doc__)
        return 2
    findings = lint(args, base)
    for fd in findings:
        tag = "warning" if fd.warning else "violation"
        print("%s:%d: %s: %s [%s]" % (fd.path, fd.line, fd.rule, fd.reason, tag))
    violations = sum(1 for fd in findings if not fd.warning)
    print("%d findings: %d violations, %d advisory warnings"
          % (len(findings), violations, len(findings) - violations))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
