#!/usr/bin/env python3
"""Nim fixture-consumer linter, deterministic checks over the Nim fixture
consumers (q_bf16/, q_exl3/, and the tests-root *.nim modules).

Doc compliance comes from lint_docs.py, imported as the single
source of truth for the linter family.


Rule table (rule | trigger | severity):

| rule-id | trigger | severity |
|---|---|---|
| assert-home | an assert* proc defined outside harness.nim: a cross-module overload bypasses the instrument, the harness = the only assert* home (noqa exempt) | counted |\n| harness-export | harness/harness.nim exports a proc outside the blessed instruments: only assertStats, assertArgMax carry * (noqa exempt) | counted |\n| global-var | a module-level var declaration in a consumer Nim file: hidden mutable state, state lives with the caller (compileTime registries exempt, noqa exempt) | counted |
| grid-name | a proc name carries grid in the transformers repo: the CUDA launch grid = the collision, name the ulp quantity (noqa exempt) | counted |
| nested-proc | an indented proc, func, iterator, template, macro, or converter declaration in a fixture consumer, a file naming the frame reader or the instruments: the consumer file hosts only main, the repetitive = standardized helpers in layer_utils.nim (noqa exempt; harness.nim, layer_utils.nim and t_harness_selftest.nim exempt) | counted |
| raw-raise | a fixture consumer names HarnessCheckError or calls newException: the consumer never raises, the harness assert* family = the only failure mechanism (noqa exempt; t_harness_selftest.nim exempt, catching the raise = its fixtures' job) | counted |
| helper-raise | newException, raise, HarnessCheckError, or quit( in a scanned tests file, suite files only when they reference the fixtures: a helper file carries enforcement power, the only authorized instruments are assertStats and assertArgMax, raising outside the harness core = bypassing the harness rules (noqa exempt; harness/harness.nim, the tests/harness.nim IO shim, harness/select_device.nim, t_harness_selftest.nim exempt) | counted |
| helper-instrument | an assert*/check*/verify*/ensure*/require* or .forward( call in a NON-suite tests file: the helper file holds the instruments, only main may call assert*/check*/forward, helpers build and return values (noqa exempt; harness/harness.nim, harness/select_device.nim, t_harness_selftest.nim, pytttransformers.nim exempt) | counted |
| nim-assert | a suite file uses Nim's assert statement: compiled out in release, the enforcement = the harness assert* procs which raise in every build (noqa exempt; t_harness_selftest.nim exempt) | counted |
| exactness-vocab | a comment or string literal says byte-exact, byte-identical, bit-exact, or bit-identical: the instruments verify, bit equality = the codec payload contract only (noqa exempt) | counted |\n| support-word | a comment or string literal says support: semantic collision with top-32 sets, softmax mass, and loader fixtures, name the actual thing (noqa exempt) | counted |
| assert-banned | an enforcement call outside the target allowlist: the harness
  assert procs (BANNED_ASSERTS), or any check*/verify*/ensure*/require* call | counted |
| assert-unknown | an assert* call outside the harness family and outside BANNED_ASSERTS | counted |
| raw-assert | a raw doAssert in a suite file | counted |
| check-proc | a check*/verify*/ensure*/require* enforcement proc defined outside tests/harness | counted |
| setup-utils | a setup* proc defined in a consumer file, the inline copy tests/layer_utils.nim replaces | counted |
| const-literal | a numeric or version const in a suite const block (filepath consts only) | counted |
| header-cap | the module doc block of a suite file over 10 content lines: tables, diagrams, bullet and numbered points, air lines, and the mandatory run command do not count | counted |
| entry-point | not exactly one flat proc main() with a when isMainModule dispatcher in a suite file | counted |
| only-main | a top-level proc other than main in a fixture consumer: the suite hosts only main, the repetitive = standardized helpers in layer_utils.nim | counted |
| section-framework | a run*Test* section wrapper defined or called in a suite file, runCppTest exempt: the repository test entry wrapper | counted |
| pass-emission | a PASS or PASSED string emitted in a suite file | counted |
| try-discard | an except branch that only discards, or a discard inside try/except, in a suite file | counted |
| proc-spacing | a top-level proc-family definition with no blank line before or after it | counted |
| comma-spacing | a comma outside a string literal or comment carrying no space or newline after it | counted |
| (lint_docs rules) | doc compliance over every scanned file, the wall rule included | per lint_docs.py |

Golden rules:
- ## docs serve API users, # comments serve maintainers and auditors
- the contract states what the code does, its preconditions, its invariants, never the journey
- every sentence stands readable to a fresh clone of the repository
- prefer bullets, tables, and diagrams over dense prose runs
- both skill files are the standard, read them first, they are always forgotten:
  - the repo skill .agents/skills/writing-docs/ (SKILL.md, REFERENCE.md, EXAMPLES.md)
  - the global skill ~/.pi/agent/skills/writing-code-doc/ (SKILL.md, references/REFERENCE.md)

- every .nim file under q_bf16/ and q_exl3/ scans, recursive
- the direct-child .nim files of the tests root scan
- the tests root is the nearest ancestor holding a q_bf16 and a harness directory
- suite files (t_ or test_ name prefix) take every rule above
- other consumer files (support modules beside the suites) get the def-shape
  rules (check-proc, setup-utils) plus doc compliance
- run `python3 lint_nim_fixtures_consumers.py <files-or-dirs>...` over a tree
- autofix the mechanical doc classes with `--fix` ahead of the file list
- the dogfood scan is `python3 lint_nim_fixtures_consumers.py
  lint_nim_fixtures_consumers.py`, it reports clean
- one finding per line, `path:line: rule-id: reason`, sorted by path and line
- advisory findings print with a [warning] tag and never set the exit code
- the lint pass never edits files, --fix mode runs the lint_docs
  mechanical autofix over the collected consumer files first
- exit 0 means clean, in --fix mode also no mechanical doc finding left
- exit 1 means at least one counted finding or mechanical doc finding left
- exit 2 means the lint_docs.py import is missing
"""

import re
import sys
from pathlib import Path

_FILE = Path(__file__).resolve()


def _skill_root():
    """Returns the repo root holding .agents/skills/writing-docs/tools,
    walking up from this file, or None."""
    d = _FILE.parent
    while d.parent != d:
        if (d / ".agents" / "skills" / "writing-docs" / "tools" / "lint_docs.py").is_file():
            return d
        d = d.parent
    return None


_SKILL_ROOT = _skill_root()
if _SKILL_ROOT is None:
    # One source of truth for the doc rules, a missing lint_docs.py is
    # a hard error, never a silent skip of the doc checks.
    sys.stderr.write(
        "lint_nim_fixtures_consumers: fatal: .agents/skills/writing-docs/tools/lint_docs.py "
        "not found above %s\n" % _FILE)
    raise SystemExit(2)
sys.path.insert(0, str(_SKILL_ROOT / ".agents" / "skills" / "writing-docs" / "tools"))
import lint_docs  # noqa  # the sys.path setup runs above, E402 silent

Finding = lint_docs.Finding

# The target allowlist, the only assert* calls a suite may call,
# both imported from tests/harness.
TARGET_ALLOWLIST = ("assertStats", "assertArgMax")

# The harness assert procs the target allowlist excludes, plus every
# check* call. A suite call outside both sets signals assert-unknown
# for that line.
BANNED_ASSERTS = ("assertAllClose", "assertMatchRate", "assertWithinBudget",
              "assertDescriptors", "assertStatsChainBand",
              "assertChainMeanDrift", "assertChainCheckpoint",
              "assertConvEvalOrder", "assertEvalOrderOutput",
              "assertSsmEvalOrder", "assertProjection", "assertClose",
              "assertTorchStamp")

PROC_RE = re.compile(r"^(?:proc|func|iterator|template|macro|converter)\s+\*?\s*(\w+)")
NESTED_PROC_RE = re.compile(
    r"\s+(?:proc|func|iterator|template|macro|converter)\s+\*?(\w+)")

# The consumer files the nested-proc rule leaves alone: the harness family,
# the standardized helper home, and the harness's own white-box selftest.
CONSUMER_DEF_EXEMPT = ("harness.nim", "layer_utils.nim", "t_harness_selftest.nim")

# The layer type names with a load* constructor in deserialization.nim form
# the sanctioned construction path, a direct build in a suite bypasses the loaders.
LAYER_TYPES = ("Linear", "RmsNorm", "RmsNormOne", "RmsNormGated", "Embedding",
               "GatedDenseFFN", "GatedBlockSparseFFN", "GatedDeltaNet",
               "RopeGQAttention", "RopeElementWiseGatedAttention", "LMHead")
LAYER_INIT_RE = re.compile(
    r"\b(?:" + "|".join(LAYER_TYPES) + r")(?:\[[^\]]*\])?\.init\s*\(")
CALL_RE = re.compile(r"\b(\w+)\s*\(")
MAIN_GUARD_RE = re.compile(r"^when\s+isMainModule\s*:")
SECTION_RE = re.compile(r"\brun(?!Cpp)\w*Test\w*\b")
PASS_STRING_RE = re.compile(r'"[^"]*\bPASS(?:ED)?\b[^"]*"')
CONST_BLOCK_RE = re.compile(r"^const\b")
NUMERIC_CONST_RE = re.compile(r"^\s*\*?\s*(\w+)\s*[^=]*=\s*(-?[\dxX][\w.xX+-]*)")
VERSION_CONST_RE = re.compile(r'^\s*\*?\s*(\w+)\s*[^=]*=\s*"[0-9][0-9xX.]*"')
EXCEPT_DISCARD_RE = re.compile(r"^except[^:]*:\s*discard\b")
BARE_ASSERT_RE = re.compile(r"\bassert\s+\S")
BARE_DOASSERT_RE = re.compile(r"\bdoAssert\s+\S")


def strip_strings(line):
    """Returns the line with double-quoted string literals removed, so call
    shapes inside string texts never read as calls.

    Quote-pairing only, adequate for suite source lines.
    """
    out = []
    in_str = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == '"' and line[i + 1:i + 2] == '"':
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(" ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def strip_comment(line):
    """Returns the line with a trailing # comment removed, keeping # inside
    string literals. Quote-counting, adequate for suite source lines."""
    in_str = None
    for i, ch in enumerate(line):
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "#":
            return line[:i]
    return line


def tests_root_for(path):
    """Returns the tests root containing path, or None for no root.

    - the root is the nearest ancestor holding a q_bf16 directory
    - the same ancestor holds the harness directory
    """
    cur = Path(path).resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        if (cur / "q_bf16").is_dir() and (cur / "harness").is_dir():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def collect_files(root):
    """Returns the scanned .nim files under a tests root, the direct children
    of the root plus everything under q_bf16/ and q_exl3/, sorted, deduped."""
    files = list(root.glob("*.nim"))
    files.extend(root.glob("harness/*.nim"))
    for area in ("q_bf16", "q_exl3"):
        if (root / area).is_dir():
            files.extend((root / area).rglob("*.nim"))
    return sorted(set(files))


def is_suite_file(path):
    """Returns True for suite files, the t_ or test_ name prefix.

    Suite files take the entry-point and enforcement-call rules.
    """
    return path.name.startswith(("t_", "test_"))

FIXTURE_CONSUMER_RE = re.compile(
    r"zstdReadFixture|harness/harness|tests/harness|assertStats|assertArgMax")

def is_fixture_consumer(lines):
    """Returns True when a suite file names the frame reader, the instrument
    module, or the instruments: the consumer law governs fixture readers,
    a property suite built from synthetic stimulus sits outside it."""
    return bool(FIXTURE_CONSUMER_RE.search("\n".join(lines)))


HARNESS_EXPORTED = {"assertStats", "assertArgMax"}
ASSERT_DEF_RE = re.compile(r"\bproc\s+assert\w*", re.IGNORECASE)
HARNESS_EXPORT_RE = re.compile(r"\bproc\s+(\w+)\*")

def check_assert_home(path, lines, findings):
    """Fires the counted assert-home rule over consumer files: assert*
    procs are defined only in harness.nim. A same-name definition
    elsewhere = a cross-module overload that silently bypasses the
    instrument the suite meant to call."""
    if path.name == "harness.nim":
        return
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("#") or "noqa" in s:
            continue
        m = ASSERT_DEF_RE.search(s)
        if m:
            findings.append(Finding(
                path, i, "assert-home",
                "an assert* proc is defined outside harness.nim: a "
                "cross-module overload bypasses the instrument, the "
                "harness = the only assert* home"))

def check_harness_exports(path, lines, findings):
    """Fires the counted harness-export rule: harness.nim exports only
    the blessed instruments assertStats and assertArgMax, every other
    proc = private."""
    if not str(path).endswith("harness/harness.nim"):
        return
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("#") or "noqa" in s:
            continue
        m = HARNESS_EXPORT_RE.search(s)
        if m and m.group(1) not in HARNESS_EXPORTED:
            findings.append(Finding(
                path, i, "harness-export",
                "harness.nim exports '%s': only assertStats, assertArgMax "
                "carry the export marker, every other proc = private"
                % m.group(1)))

GLOBAL_VAR_RE = re.compile(r"^var ", re.MULTILINE)

def check_global_var(path, text, findings):
    """Fires the counted global-var rule over consumer files: no
    module-level mutable var exists in the transformers Nim, state
    lives in the caller's hands."""
    for m in GLOBAL_VAR_RE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        raw = text.splitlines()[line - 1]
        if "noqa" in raw:
            continue
        context = "\n".join(text.splitlines()[max(0, line - 3):line + 2])
        if "{.compileTime.}" in context or "compileTime" in raw:
            continue
        findings.append(Finding(
            path, line, "global-var",
            "a module-level var = hidden mutable state, state lives "
            "with the caller"))

BANNED_ULP_TENSOR_RE = re.compile(
    r"\bTensor\b|libtorch|\bF\.[A-Z]")

BANNED_GRID_NAME_RE = re.compile(
    r"\bproc\s+\w*grid\w*\*", re.IGNORECASE)

def check_grid_name(path, lines, findings):
    """Fires the counted grid-name rule over consumer files: no proc name
    carries grid in the transformers repo, the CUDA launch grid = the
    collision, the ulp vocabulary names the ulp procs."""
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("#") or "noqa" in s:
            continue
        if BANNED_GRID_NAME_RE.search(s):
            findings.append(Finding(
                path, i, "grid-name",
                "a proc name carries grid: the CUDA launch grid = the "
                "collision, name the ulp quantity directly"))

BANNED_EXACTNESS_RE = re.compile(
    r"\bbyte[-_ ]exact\b|\bbyte[-_ ]identical\b|\bbit[-_ ]exact\b|\bbit[-_ ]identical\b",
    re.IGNORECASE)

BANNED_SUPPORT_RE = re.compile(r"\bsupport\b", re.IGNORECASE)

def check_banned_support(path, lines, findings):
    """Fires the counted support-word rule over Nim comment lines and
    string literals: the word collides with top-32 sets, softmax mass,
    and loader fixtures, name the actual thing."""
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not (s.startswith("#") or s.startswith("##") or '"' in s):
            continue
        body = s.lstrip("#").strip()
        if not body or "noqa" in s:
            continue
        m = BANNED_SUPPORT_RE.search(body)
        if m:
            findings.append(Finding(
                path, i, "support-word",
                "banned word 'support': semantic collision, name the top-32 "
                "set, the tail probability, or the loader fixture explicitly"))

def check_banned_exactness(path, lines, findings):
    """Fires the counted exactness-vocab rule over Nim comment lines and
    string literals: byte/bit identity is banned as a pass criterion, the
    instruments verify. The codec payload contract carries the ruled
    exception behind a noqa."""
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not (s.startswith("#") or s.startswith("##") or '"' in s):
            continue
        body = s.lstrip("#").strip()
        if not body or "noqa" in s:
            continue
        m = BANNED_EXACTNESS_RE.search(body)
        if m:
            findings.append(Finding(
                path, i, "exactness-vocab",
                "banned exactness vocabulary '%s': bit/byte exactness violates the harness goal, robustness and portability across platforms; verify by the instruments" % m.group(0)))

def check_defs(path, lines, findings, consumer=True):
    """Fires the def-shape rules over one consumer file:
    - check* procs live in tests/harness, nowhere else
    - setup* procs live in tests/layer_utils.nim, nowhere else
    - any indented proc, func, iterator, template, macro, or converter
      declaration fires nested-proc, a fixture consumer file hosts only main
    The prefix rules read column 0 only, and each home skips its own file."""
    homes = {"setup": "layer_utils.nim", "check": "harness.nim"}
    for i, raw in enumerate(lines, 1):
        if "noqa" in raw:
            continue
        code = strip_comment(raw)
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip())
        nm = NESTED_PROC_RE.match(code)
        if indent > 0 and nm and path.name not in CONSUMER_DEF_EXEMPT \
                and consumer:
            findings.append(Finding(
                path, i, "nested-proc",
                "`%s`: a fixture consumer file hosts only main, the "
                "repetitive = standardized helpers in layer_utils.nim"
                % nm.group(1)))
        m = PROC_RE.match(code)
        if not m:
            continue
        name = m.group(1)
        if name.startswith(("check", "verify", "ensure", "require")):
            if path.name == homes["check"]:
                continue
            findings.append(Finding(
                path, i, "check-proc",
                "proc `%s`: enforcement procs live in tests/harness"
                % name))
        elif name.startswith("setup"):
            if path.name == homes["setup"]:
                continue
            findings.append(Finding(
                path, i, "setup-utils",
                "proc `%s`: setup functions live in tests/layer_utils.nim, "
                "this is the inline copy it replaces" % name))


SETUP_LAYER_SIG_RE = re.compile(
    r"\s*\*?\s*\(\s*cfg:\s*JsonNode\s*,\s*T:\s*typedesc\[")


def check_setup_home(path, lines, findings):
    """Fires the setup-home shape rule over tests/layer_utils.nim.
    - the generic layer dispatch carries the bare overloaded setup name,
      cfg the first argument, the layer typedesc second
    - a setup-named proc with the constructor signature must stay bare,
      the standardized replay helpers carry the setup* prefix with their
      own signatures"""
    if path.name != "layer_utils.nim":
        return
    joined = "\n".join(lines)
    for m in re.finditer(r"(?m)^(?:proc|func|iterator|template|macro|converter)\s+\*?\s*(\w+)", joined):
        name = m.group(1)
        line_no = joined[:m.start()].count("\n") + 1
        sig = joined[m.end():m.end() + 400]
        if name != "setup" and SETUP_LAYER_SIG_RE.match(sig):
            findings.append(Finding(
                path, line_no, "setup-shape",
                "proc `%s`: the layer constructors carry the bare overloaded "
                "name setup, cfg first, the layer type second" % name))
        elif name == "setup" and not SETUP_LAYER_SIG_RE.match(sig):
            findings.append(Finding(
                path, line_no, "setup-shape",
                "proc `setup`: the signature opens (cfg: JsonNode, "
                "T: typedesc[<Layer>], ...), cfg first, the layer type second"))


BANNED_SUITE_RAISE_RE = re.compile(
    r"\bnewException\b|\bHarnessCheckError\b", re.IGNORECASE)

def check_raw_raise(path, lines, findings):
    """Fires the counted raw-raise rule over suite files only: a fixture
    consumer raises nothing itself, the only failure mechanism = the
    harness assert* family. Raising HarnessCheckError directly or
    newException at all = bypassing the instruments, and catching them =
    eating the harness verdict. Harness-internal files stay exempt, the
    providers must raise, and the selftest stays exempt, catching the
    harness raise = its expect-raise fixtures' job."""
    if path.name == "t_harness_selftest.nim":
        return
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("#") or "noqa" in s:
            continue
        m = BANNED_SUITE_RAISE_RE.search(s)
        if m:
            findings.append(Finding(
                path, i, "raw-raise",
                "a fixture consumer never raises: the only failure mechanism "
                "= the harness assert* family, '%s' = bypassing the "
                "instruments" % m.group(0)))

HELPER_RAISE_RE = re.compile(
    r"\bnewException\b|\braise\s|\bHarnessCheckError\b|\bquit\s*\(")

def check_helper_raise(path, lines, findings):
    """Fires the counted helper-raise rule over every scanned tests file:
    - newException, a raise statement, HarnessCheckError, or quit grants a
      helper file enforcement power
    - the only authorized instruments are assertStats and assertArgMax,
      raising outside the harness = bypassing the harness rules
    - the harness core stays exempt, the providers must raise
    - the IO shim tests/harness.nim stays exempt, the frame reader raises
      on an empty frame, IO is separate from the pure harness"""
    if str(path).replace("\\", "/").endswith("harness/harness.nim"):
        return
    if path.name == "harness.nim" and path.parent.name == "tests":
        return
    if path.name in ("select_device.nim", "t_harness_selftest.nim"):
        return
    for i, raw in enumerate(lines, 1):
        if "noqa" in raw:
            continue
        m = HELPER_RAISE_RE.search(raw)
        if m:
            findings.append(Finding(
                path, i, "helper-raise",
                "helper file carries enforcement power: the only authorized "
                "instruments are assertStats and assertArgMax, raising "
                "outside the harness = bypassing the harness rules, '%s'"
                % m.group(0)))

HELPER_INSTRUMENT_RE = re.compile(
    r"\b(?:assert|check|verify|ensure|require)\w*\s*\(|\.forward\s*\(")

def check_helper_instrument(path, lines, findings):
    """Fires the counted helper-instrument rule over every scanned tests
    file that is not a suite:
    - an assert*/check*/verify*/ensure*/require* call, or a .forward( call,
      in a helper file means the helper drives the model or judges the
      result, the instruments and the inference loop live in main only
    - the exemption list mirrors helper-raise, the harness core, the device
      selector and the harness selftest keep their instruments
    - the nimpy binding pytttransformers.nim stays exempt, its forward is
      the Python-facing API, not a suite-driving helper"""
    if is_suite_file(path):
        return
    if str(path).replace("\\", "/").endswith("harness/harness.nim"):
        return
    if path.name in ("select_device.nim", "t_harness_selftest.nim",
                     "pytttransformers.nim"):
        return
    for i, raw in enumerate(lines, 1):
        if "noqa" in raw:
            continue
        code = strip_strings(strip_comment(raw))
        if not code.strip() or code.strip().startswith("#"):
            continue
        m = HELPER_INSTRUMENT_RE.search(code)
        if m:
            findings.append(Finding(
                path, i, "helper-instrument",
                "helper file holds the instruments: only main calls "
                "assert*/check*/forward, helpers build and return, '%s'"
                % m.group(0)))

NIM_ASSERT_RE = re.compile(r"\bassert\b", re.IGNORECASE)

def check_nim_assert(path, lines, findings):
    """Fires the counted nim-assert rule over suite files: Nim's assert
    statement = compiled out in release builds, a fixture consumer that
    enforces through assert tests nothing in the release replay. The
    enforcement = the harness assert* procs, which reject through the
    instrument in every build. assertAllClose/assertArgMax/assertStats =
    unaffected, the word boundary stops inside the camel case."""
    if path.name == "t_harness_selftest.nim":
        return
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("#") or "noqa" in s:
            continue
        if NIM_ASSERT_RE.search(s):
            findings.append(Finding(
                path, i, "nim-assert",
                "Nim assert = compiled out in release: the suite enforces "
                "through the harness assert* family, which raises in every "
                "build"))

def check_consts(path, lines, findings):
    """Fires the counted const-literal rule over one suite file.

    Const blocks hold filepath consts only:
    - geometry and version literals are banned
    - the suite reads the handed config.json or the fixture manifest
    """
    in_block = False
    for i, raw in enumerate(lines, 1):
        code = strip_comment(raw)
        if CONST_BLOCK_RE.match(code):
            in_block = True
            continue
        if in_block:
            if not code.strip():
                continue
            if code[0] not in " \t":
                in_block = False
                continue
        else:
            continue
        m = NUMERIC_CONST_RE.match(code)
        if m and re.match(r"^-?\d", m.group(2)):
            findings.append(Finding(
                path, i, "const-literal",
                "const `%s = %s`: const blocks hold filepath consts only, "
                "read the value from the handed config or the fixture "
                "manifest" % (m.group(1), m.group(2))))
        m = VERSION_CONST_RE.match(code)
        if m:
            findings.append(Finding(
                path, i, "const-literal",
                "const `%s`: version consts are banned, read the value from "
                "the handed config or the fixture manifest" % m.group(1)))


def check_entry_point(path, lines, findings, consumer=True):
    """Fires the entry-point and section rules over one suite file.

    - exactly one flat proc main()
    - one when isMainModule dispatcher
    - no run*Test* section wrapper defined or called
    - no PASS string emitted
    - no top-level function besides main
    - no direct layer construction, the deserialization loaders build layers
    """
    mains = []
    guards = []
    for i, raw in enumerate(lines, 1):
        code = strip_comment(raw)
        m = PROC_RE.match(code)
        if m:
            if m.group(1) == "main":
                mains.append(i)
            elif consumer:
                findings.append(Finding(
                    path, i, "only-main",
                    "top-level `%s`: a fixture consumer file hosts only main, "
                    "the repetitive = standardized helpers in layer_utils.nim"
                    % m.group(1)))
        if MAIN_GUARD_RE.match(code):
            guards.append(i)
        lm = LAYER_INIT_RE.search(code)
        if lm:
            findings.append(Finding(
                path, i, "layer-init",
                "direct layer construction `%s`: the deserialization loaders "
                "build every layer, read the weights through load*" % lm.group(0)))
        if SECTION_RE.search(code):
            findings.append(Finding(
                path, i, "section-framework",
                "run*Test* section wrapper: one flat proc main() per suite "
                "file, no section framework"))
        if PASS_STRING_RE.search(code):
            findings.append(Finding(
                path, i, "pass-emission",
                "PASS string emitted: the suite result is the exit code, "
                "never printed pass counters"))
    if not mains:
        findings.append(Finding(
            path, 1, "entry-point",
            "no proc main(): one flat main per suite file"))
    for line in mains[1:]:
        findings.append(Finding(
            path, line, "entry-point", "second proc main(): keep exactly one"))
    if not guards:
        findings.append(Finding(
            path, 1, "entry-point",
            "no `when isMainModule` dispatcher: importing the suite module "
            "must never run it"))


def check_calls(path, lines, findings):
    """Fires the enforcement-call rules over one suite file:
    - assert-banned, the excluded procs plus every check*, verify*,
      ensure*, and require* call, counted
    - assert-unknown, any other assert* call outside the target allowlist
    - raw-assert, a raw doAssert
    - try-discard, an except branch that only discards, or a discard
      inside try/except, error eating around a check is banned
    - t_harness_selftest.nim = the harness's own white-box suite, it
      exercises the internal check procs and the rules stay off
    """
    if path.name == "t_harness_selftest.nim":
        return
    try_stack = []
    for i, raw in enumerate(lines, 1):
        s = strip_comment(raw)
        stripped = s.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(s) - len(s.lstrip())
        while try_stack and try_stack[-1][1] >= indent:
            try_stack.pop()
        if EXCEPT_DISCARD_RE.match(stripped):
            findings.append(Finding(
                path, i, "try-discard",
                "inline except-discard: error eating around a check is "
                "banned, let the rejection surface"))
            continue
        if stripped == "try:":
            try_stack.append(("try", indent))
            continue
        if stripped.startswith(("except", "finally")):
            try_stack.append(("exc", indent))
            continue
        if stripped == "discard" or stripped.startswith("discard "):
            if any(kind == "exc" for kind, _ in try_stack):
                findings.append(Finding(
                    path, i, "try-discard",
                    "discard inside try/except: error eating around a check "
                    "is banned, let the rejection surface"))
            continue
        bare = strip_strings(stripped)
        for call in CALL_RE.findall(bare):
            if call == "doAssert":
                findings.append(Finding(
                    path, i, "raw-assert",
                    "raw doAssert: enforcement goes through the harness "
                    "family"))
            elif call in BANNED_ASSERTS or call.startswith(
                    ("check", "verify", "ensure", "require")):
                findings.append(Finding(
                    path, i, "assert-banned",
                    "banned enforcement call `%s`: only assertStats and "
                    "assertArgMax may enforce" % call))
            elif call == "assert" or (call.startswith("assert")
                                      and call not in TARGET_ALLOWLIST):
                findings.append(Finding(
                    path, i, "assert-unknown",
                    "assert call `%s` outside the harness family and "
                    "outside the target allowlist" % call))
        if BARE_DOASSERT_RE.search(bare):
            findings.append(Finding(
                path, i, "raw-assert",
                "raw doAssert: enforcement goes through the harness family"))
        elif BARE_ASSERT_RE.search(bare):
            findings.append(Finding(
                path, i, "assert-unknown",
                "bare assert: enforcement goes through the harness family"))


HEADER_CAP = 10


def is_command_body(body, prev_ended_backslash):
    """Command-line shape test for a doc-block body line.

    - Args:
      body is the doc-block line without its `##` prefix
      prev_ended_backslash says whether the previous body line ended
      with a backslash continuation
    - Returns:
      True for a run-command line, which stays mandatory,
      kept uncounted against the header cap
    """
    if prev_ended_backslash:
        return True
    return bool(lint_docs.RUN_CMD_RE.search(body))


def check_header_cap(path, text, findings):
    """A counted rule, a suite file module doc block carries at most 10 content lines.

    - Args:
      path is the suite file under check, text is the full file
      text (findings append in place, nothing returns)

    Not counted against the cap:

    - air lines, tables, diagrams, bullets and numbered points
    - run-command lines, mandatory per the doc-comment linter's test-header-command check
    """
    count = 0
    first = None
    prev_backslash = False
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line.startswith("##"):
            if first is not None:
                break
            continue
        body = line[2:].strip()
        if first is None:
            if not body:
                continue
            first = i
        elif not body:
            continue
        if (lint_docs.is_table(body) or lint_docs.is_diagram(body)
                or lint_docs.BULLET_RE.match(body)):
            prev_backslash = body.endswith("\\")
            continue
        if is_command_body(body, prev_backslash):
            prev_backslash = body.endswith("\\")
            continue
        prev_backslash = body.endswith("\\")
        count += 1
    if first is None:
        return
    if count > HEADER_CAP:
        findings.append(Finding(
            path, first, "header-cap",
            "module doc block carries %d content lines, the cap is %d"
            % (count, HEADER_CAP)))


def check_proc_spacing(path, lines, findings):
    """One blank line before and after every top-level proc-family def.

    Column-0 defs only, nested and indented defs stay out. The before
    line may be absent at the file head, the rule reads the separation
    between defs, not the file opening.
    """
    tops = [i for i, raw in enumerate(lines)
            if raw.strip() and not raw[0] in " \t"]
    for k, i in enumerate(tops):
        m = PROC_RE.match(strip_comment(lines[i]))
        if not m:
            continue
        name = m.group(1)
        if i > 0 and lines[i - 1].strip():
            findings.append(Finding(
                path, i + 1, "proc-spacing",
                "proc `%s`: no blank line before the definition" % name))
        end = tops[k + 1] - 1 if k + 1 < len(tops) else len(lines) - 1
        while end > i and not lines[end].strip():
            end -= 1
        if end + 1 < len(lines) and lines[end + 1].strip():
            findings.append(Finding(
                path, end + 2, "proc-spacing",
                "proc `%s`: no blank line after the definition" % name))


def check_comma_spacing(path, lines, findings):
    """Fires the comma-spacing rule over one consumer file: every comma
    outside a string literal or a comment carries a space or the line
    end after it, `(a,b)` is a finding, `(a, b)` is not."""
    for i, raw in enumerate(lines, 1):
        code = strip_comment(raw)
        in_str = None
        j = 0
        while j < len(code):
            ch = code[j]
            if in_str:
                if ch == "\\":
                    j += 2
                    continue
                if ch == in_str:
                    in_str = None
            elif ch in "\"'":
                in_str = ch
            elif ch == ",":
                if j + 1 < len(code) and code[j + 1] not in " \t":
                    findings.append(Finding(
                        path, i, "comma-spacing",
                        "comma carries no space after it"))
            j += 1


def scan(path, text, findings):
    """Runs every rule over one consumer file, appending to findings."""
    for fd in lint_docs.lint_text(text, str(path)):
        findings.append(Finding(fd.path, fd.line, fd.rule, fd.reason, fd.warning))
    lines = text.splitlines()
    consumer = is_fixture_consumer(lines)
    check_defs(path, lines, findings, consumer)
    check_proc_spacing(path, lines, findings)
    check_setup_home(path, lines, findings)
    check_comma_spacing(path, lines, findings)
    check_banned_exactness(path, lines, findings)
    check_banned_support(path, lines, findings)
    check_grid_name(path, lines, findings)
    check_global_var(path, text, findings)
    check_assert_home(path, lines, findings)
    check_harness_exports(path, lines, findings)
    if consumer or not is_suite_file(path):
        check_helper_raise(path, lines, findings)
    check_helper_instrument(path, lines, findings)
    if not is_suite_file(path):
        return
    check_header_cap(path, text, findings)
    if consumer:
        check_raw_raise(path, lines, findings)
        check_nim_assert(path, lines, findings)
    check_consts(path, lines, findings)
    check_entry_point(path, lines, findings, consumer)
    check_calls(path, lines, findings)


def lint(paths):
    """Lints the .nim files the caller names, without touching anything else.

    - a file argument is scanned as itself, no tests-root expansion
    - a directory argument expands to the consumer files under its tests root,
      the whole tree when the directory is the root
    - a non-.nim file argument, or a path with no tests root, gets a note
      on stderr and is skipped

    Returns the findings sorted by path, line, and rule id.
    """
    findings = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix != ".nim":
                sys.stderr.write("lint_nim_fixtures_consumers: not a .nim file, "
                                 "skipping %s\n" % p)
                continue
            scan(path, path.read_text(encoding="utf-8", errors="replace"),
                 findings)
            continue
        root = tests_root_for(p)
        if root is None:
            sys.stderr.write("lint_nim_fixtures_consumers: no tests root found for "
                             "%s, skipping\n" % p)
            continue
        for f in collect_files(root):
            scan(f, f.read_text(encoding="utf-8", errors="replace"), findings)
    findings.sort(key=lambda x: (str(x.path), x.line, x.rule))
    return findings


def main(argv):
    """Runs the CLI on a file and directory list, with --fix running the lint_docs
    mechanical autofix over the collected consumer files first. Returns the process
    exit code (0 clean, 1 findings, 2 failure)."""
    args = argv[1:]
    fix_mode = "--fix" in args
    args = [a for a in args if a != "--fix"]
    mechanical_left = 0
    if fix_mode and args:
        files = set()
        for a in args:
            path = Path(a)
            if path.is_file() and path.suffix == ".nim":
                files.add(path)
                continue
            root = tests_root_for(a)
            if root is not None:
                files.update(collect_files(root))
        mechanical_left = sum(lint_docs.fix_one_reported(f)
                              for f in sorted(files))
    if not args:
        print(__doc__)
        return 2
    findings = lint(args)
    for fd in findings:
        tag = "warning" if fd.warning else "violation"
        print("%s:%d: %s: %s [%s]" % (fd.path, fd.line, fd.rule, fd.reason, tag))
    violations = sum(1 for fd in findings if not fd.warning)
    print("%d findings: %d violations, %d advisory warnings"
          % (len(findings), violations, len(findings) - violations))
    return 1 if (violations or mechanical_left) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
