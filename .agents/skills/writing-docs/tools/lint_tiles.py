#!/usr/bin/env python3
"""Register-tile linter for the ceramic element-wise kernel layer.

Scans .nim kernel sources and flags the hand-rolled elementwise patterns
the tile_algebra primitives already replace, one finding per site, over
these roots:

- workspace/positron/src/kernels/
- workspace/positron/src/mega_kernels/
- workspace/ceramic/src/

| rule-id           | trigger                                                                                          | severity |
| ----------------- | ------------------------------------------------------------------------------------------------ | -------- |
| frag-walk         | a `{.device.}`/`{.global.}` proc body reads or writes `.frags[..][..].frag[..]` without a marker | counted  |
| tile-op-miss      | a frag assignment whose right side is single-frag arithmetic expressible as existing tile ops    | counted  |
| lane-cell         | manual `crd2idx` + `cell mod/div` lane-to-cell decomposition outside the tile_algebra primitives | counted  |
| scalar-extract    | raw `.data[0]` scalar extraction from a row vector outside the tile_algebra primitives           | counted  |
| magic-dim         | a bare 8/16/32/64/128/256 literal in register-tile dims or index arithmetic                      | counted  |
| raw-emit          | a `{.emit:}` block spelling a Crucible builtin (threadgroup_barrier) by hand                     | counted  |
| math-const        | a float literal that bitwise matches a shared math_consts value                                  | counted  |
| kernel-doc-shape  | an exported `{.device.}`/`{.global.}` kernel with no SDPA-form doc block                         | counted  |
| hash-above-proc   | a `#` comment sits immediately above a proc, func, or template definition                        | counted  |
| divider-space     | a section divider (`# ─── ... ───`) without a blank line before or after it                      | counted  |
| one-liner         | a proc, func, or template whose body is one code line, the wrapper shape stays evident at review | advisory |
| explicit-generics | a call site spells generic arguments the compiler infers from the value arguments                | counted  |

Module-scope exemptions, where a frag walk IS the tile implementation:

- workspace/ceramic/src/tile_algebra/ (all modules)
- workspace/positron/src/kernels/ceramic/tile_io_rows.nim
- workspace/positron/src/kernels/ceramic/tile_widen.nim

Exemption behavior:

- those paths skip frag-walk, tile-op-miss, and kernel-doc-shape
- lane-cell and magic-dim still run there
- only tile_algebra silences lane-cell, the shared `laneCellOf`
  helper belongs in that module

Allowlist marker, proc scope:

- a `# tiles-allow <what> needs <primitive>` line in a proc body or up
  to 6 lines above its header exempts that one proc from frag-walk
- the marker names the primitive the walk waits for, so a legitimate
  lane machine (butterfly cores, rope lane shuffles) stays auditable
  without silencing the elementwise defeats around it
- lane-cell and scalar-extract report sites as inventory for the pending
  tile_algebra primitive, never a proc that exists today

Usage (mirrors lint_docs.py):

    python3 lint_tiles.py <files-or-dirs>...
    python3 lint_tiles.py --base <commit> <files-or-dirs>...

- with no paths the three kernel roots are scanned
- with --base (or TILE_LINT_BASE in the environment), findings are
  scoped to the lines the diff from that commit adds, pre-existing
  violations a change did not touch stay out of the report

Output is one finding per line in the `path:line: rule-id: reason` shape,
sorted by path and line.

- exit 0 means no counted finding (advisory findings alone still exit 0)
- exit 1 means at least one counted finding
- exit 2 means a usage or tool failure
"""

import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_docs import Finding, added_lines, nim_block_comment_lines  # noqa: E402

KERNEL_ROOTS = (
    os.path.join("workspace", "positron", "src", "kernels"),
    os.path.join("workspace", "positron", "src", "mega_kernels"),
    os.path.join("workspace", "ceramic", "src"),
)

# Modules where a fragment walk is the implementation, not a defeat.
# Posix-normalized relative paths, matched per module.
WALK_EXEMPT = tuple(
    [os.path.join("workspace", "ceramic", "src", "tile_algebra") + os.sep,
     os.path.join("workspace", "positron", "src", "kernels", "ceramic",
                  "tile_io_rows.nim"),
     os.path.join("workspace", "positron", "src", "kernels", "ceramic",
                  "tile_widen.nim")]
)

# Only tile_algebra silences lane-cell, the shared `laneCellOf` helper
# belongs in those modules.
LANE_CELL_EXEMPT = (os.path.join("workspace", "ceramic", "src", "tile_algebra")
                    + os.sep,)

RULES = {
    "frag-walk": "a {.device.}/{.global.} proc body walks .frags[..][..].frag[..] "
                 "without an allowlist marker",
    "tile-op-miss": "single-frag arithmetic a tile_algebra op already expresses",
    "lane-cell": "manual crd2idx + mod/div lane-to-cell decomposition",
    "scalar-extract": "raw .data[0] scalar extraction from a row vector",
    "magic-dim": "a bare tile-dim literal in register-tile dims or index arithmetic",
    "raw-emit": "a {.emit:} block spelling a Crucible builtin by hand",
    "math-const": "a float literal re-spelling a shared math_consts value",
    "kernel-doc-shape": "an exported kernel with no SDPA-form doc block",
    "hash-above-proc": "a # comment sits immediately above a proc/func/template "
                       "definition, use a ## doc comment; Nim reads ##",
    "divider-space": "a section divider keeps a blank line before and after it",
    "one-liner": "a proc/func/template body is a single line, the wrapper shape "
                 "is named at review, never a violation",
    "explicit-generics": "a call site spells generic arguments the compiler "
                         "infers from the value arguments",
}

# Callable declarations the new structural rules read. The tile rules stay
# scoped to these forms, macros and iterators keep their own conventions.
CALLABLE_HEAD_RE = re.compile(
    r"^\s*(?:proc|func|template|macro|iterator|converter)\b")
CALLABLE_NAME_RE = re.compile(
    r"^\s*(?:proc|func|template|macro|iterator|converter)\s+(\*|`)?\s*"
    r"([A-Za-z_]\w*)")
RULE_KINDS = frozenset(("proc", "func", "template"))
CALL_SITE_RE = re.compile(r"\b([A-Za-z_]\w*)\[([^\[\]]+)\]\s*\(")

# Section divider, a box-drawing rule line possibly carrying a title
# (`# ─── The kernel ───`). A plain `-` never counts, prose dashes stay prose.
DIVIDER_RE = re.compile(r"^\s*#(?!#).*[─━═]{3,}")

PROC_HEAD_RE = re.compile(r"^\s*(?:proc|func|iterator|template|macro)\s+\*?")
DEVICE_RE = re.compile(r"\{\.\s*(device|global)\b")
PRAGMA_END_EQ_RE = re.compile(r"[=]\s*$")
EXPORTED_RE = re.compile(r"^\s*(?:proc|func|iterator|template|macro)\s+\w+\*")

# Fragment reference shape `x.frags[n][m].frag[v]`, indices are arbitrary expressions.
FRAG_REF_RE = re.compile(
    r"[A-Za-z_]\w*(?:\[[^\]]*\])*\.frags\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\.frag\s*\[[^\]]+\]")

FRAG_LHS_ASSIGN_RE = re.compile(
    r"(?:^|[^\w.])()(?:[A-Za-z_]\w*(?:\[[^\]]*\])*)\.frags\s*\[[^\]]+\]\s*\[[^\]]+\]\s*"
    r"\.frag\s*\[[^\]]+\]\s*=(?!=)")

ALLOW_MARKER_RE = re.compile(r"^\s*#+" r"\s*tiles-allow\b\s*(\S.*)$")
ALLOW_MARKER_LOOKBACK = 6

SCALAR_EXTRACT_RE = re.compile(r"\b[A-Za-z_]\w*\.data\s*\[\s*0\s*\]")
CRD2IDX_RE = re.compile(r"\bcrd2idx\s*\(")
CELL_MOD_DIV_RE = re.compile(r"\bcell\s+(?:mod|div)\b")

MAGIC_DIMS = frozenset((8, 16, 32, 64, 128, 256))
MAGIC_LIT_RE = re.compile(r"(?<![\w.])(8|16|32|64|128|256)(?![\w.'\"])")
RT_CALL_RE = re.compile(r"\brt_[lrwv]\s*\(")
FOR_HEADER_RE = re.compile(r"^\s*for\b")
CONST_DEF_RE = re.compile(r"^\s*const\b")
GETTER_RE = re.compile(r"\bget[MN]\(\)|getVpt\(\)")

EMIT_OPEN_RE = re.compile(r"\{\.\s*emit\s*:")
EMIT_CLOSE_RE = re.compile(r'"""')

FLOAT_LIT_RE = re.compile(
    r"(?<![\w.])([+-]?\d+\.\d+(?:[eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+)"
    r"('f32|'f64|'f)?(?![\w.'])")
CONST_FLOAT_RE = re.compile(
    r"^\s*const\s+(\w+)\*?\s*=\s*([0-9.]+(?:[eE][+-]?\d+)?)'f(?:32|64)")

DOC_PARAM_MARKERS = ("Parameters:", "Expected input", "Output:")

# Tokens a single-frag arithmetic right side may carry beyond the fragment
# references:
# - scalar identifiers and literals
# - the arithmetic and call forms the tile op surface composes
#   (add, sub, mul, exp2, convert)
ARITH_TOKEN_RE = re.compile(r"^[A-Za-z_]\w*$|^\d|^[+\-*/(),]$|^\.$")
ARITH_FUNC_NAMES = frozenset(("exp2", "to", "float32", "float16", "bfloat16"))
BITWISE_TOKEN_RE = re.compile(r"\b(?:shl|shr|and|or|xor|mod|div)\b")


def _rel(path):
    """Returns the posix-normalized repo-relative path of one file."""
    root = Path.cwd()
    try:
        return Path(os.path.relpath(str(path), root)).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _walk_exempt(path):
    """Returns True for the modules where a fragment walk is the implementation."""
    rel = _rel(path)
    return "/tile_algebra/" in rel \
        or rel.endswith(("ceramic/tile_io_rows.nim", "ceramic/tile_widen.nim"))


def _lane_cell_exempt(path):
    """Returns True for the tile_algebra modules the lane-cell check silences."""
    return "/tile_algebra/" in _rel(path)


def _strip_comment(raw):
    """Returns the code text of one raw line, `##` doc lines and trailing
    `#` comments removed, and True when the line was comment-only."""
    s = raw.strip()
    if s.startswith("##") or s.startswith("#"):
        return "", True
    m = re.search(r"\s#", raw)
    if m:
        raw = raw[:m.start()]
    return raw.rstrip(), False


def device_procs(lines):
    """Extracts the {.device.}/{.global.} procs of one file.

    Returns a list of records, one per proc.

| field      | contents                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------- |
| start, end | the 1-based inclusive line span (header start, body end)                                  |
| header     | the 0-based header line number                                                            |
| exported   | whether the proc name carries Nim's export marker                                         |
| body       | (line no, code text) pairs, comments stripped, `{.emit:}` blocks as collected string text |

    A proc spans from its header line through the signature's `=`
    terminator line, then every body line (blank, comment, or indented deeper than the header).
    """
    out = []
    i, n = 0, len(lines)
    while i < n:
        raw, _c = _strip_comment(lines[i])
        if not PROC_HEAD_RE.match(raw):
            i += 1
            continue
        head = i
        indent = len(lines[i]) - len(lines[i].lstrip())
        # Scan the signature to its `=` terminator, the {.device.}/{.global.} pragma sits inside that span.
        header_text = []
        j = i
        eq = None
        while j < n and j <= i + 25:
            code, comment_only = _strip_comment(lines[j])
            header_text.append(code)
            joined = " ".join(header_text)
            if DEVICE_RE.search(joined) and PRAGMA_END_EQ_RE.search(joined):
                eq = j
                break
            if j > i and PROC_HEAD_RE.match(lines[j].strip()) \
                    and re.match(r"^\s*(?:proc|func|iterator|template|macro)\b",
                                 lines[j]):
                break
            j += 1
        if eq is None:
            i += 1
            continue
        device = bool(DEVICE_RE.search(" ".join(header_text)))
        exported = bool(EXPORTED_RE.match(lines[head]))
        body = []
        k = eq + 1
        while k < n:
            line = lines[k]
            if not line.strip():
                body.append((k + 1, ""))
                k += 1
                continue
            if not line.lstrip().startswith("##") \
                    and len(line) - len(line.lstrip()) <= indent:
                # doc lines attach to the proc at any indent, shallower code ends the body
                break
            code, comment_only = _strip_comment(line)
            if comment_only:
                body.append((k + 1, None))  # comment line, markers read below
                k += 1
                continue
            if EMIT_OPEN_RE.search(code):
                # Collect the raw emit string for raw-emit, keep the other
                # checks off the emitted target-language text.
                emit_lines = [code]
                k += 1
                while k < n:
                    emit_lines.append(lines[k])
                    if EMIT_CLOSE_RE.search(lines[k]):
                        k += 1
                        break
                    k += 1
                body.append((k, "EMIT:" + "\n".join(emit_lines)))
                continue
            body.append((k + 1, code))
            k += 1
        out.append({"start": head + 1, "end": max(k, eq + 1), "header": head,
                    "exported": exported, "device": device, "body": body,
                    "indent": indent})
        i = max(k, eq + 1)
    return out


def _allowlisted_proc(proc, lines):
    """Returns True when one proc carries its own `tiles-allow` marker, a raw
    line in the lookback window above the header or in the proc's
    line span (body markers and emit blocks included)."""
    first = max(0, proc["header"] - ALLOW_MARKER_LOOKBACK)
    for j in range(first, min(proc["end"], len(lines))):
        if ALLOW_MARKER_RE.match(lines[j]):
            return True
    return False


def _join_rhs(lines, start_k, initial):
    """Returns one assignment's right side, the initial post-`=` text
    then every continuation line appended until the brackets balance
    (or the proc body ends)."""
    text = initial
    k = start_k
    bal = (text.count("(") + text.count("[")
           - text.count(")") - text.count("]"))
    if not text.strip():
        # an assignment line ending on `=` opens the continuation
        bal = 1
    while k < len(lines) and bal > 0:
        code = lines[k]
        if code is None:
            k += 1
            continue
        text += (" " if text else "") + code
        k += 1
        # Bracket balance over the joined text, string content ignored
        # (single-frag arithmetic carries no string literals).
        bal = (text.count("(") + text.count("[")
               - text.count(")") - text.count("]"))
    return text, k


def _frag_indices(refs):
    """Returns the set of whitespace-normalized index triples of frag refs."""
    out = set()
    for ref in refs:
        idxs = re.findall(r"\[([^\]]*)\]", ref)
        out.add(tuple("".join(i.split()) for i in idxs))
    return out


def _classify_single_frag(rhs, refs):
    """Classifies one same-frag-index arithmetic right side into the tile
    ops that express it, None when the form escapes the op surface."""
    leftover = rhs
    for ref in refs:
        leftover = leftover.replace(ref, " ")
    if re.search(r"\[", leftover) or ".data" in leftover \
            or re.search(r"\bgd\s*\(", leftover) or BITWISE_TOKEN_RE.search(leftover):
        return None
    ops = []
    if re.search(r"\bexp2\s*\(", leftover):
        ops.append("exp2")
    if re.search(r"\.to\s*\(|\.float32\b|\.float16\b|\.bfloat16\b", leftover):
        ops.append("convert")
    stars = len(re.findall(r"(?<![*/+\-\s*])\s*[*]\s*", leftover))
    slashes = len(re.findall(r"/", leftover))
    pluses = len(re.findall(r"\+", leftover))
    minuses = len(re.findall(r"-", leftover))
    if slashes and not stars:
        ops.append("mul (reciprocal form)")
    elif stars:
        ops.append("mul")
    if pluses:
        ops.append("add")
    if minuses:
        ops.append("sub")
    if not ops and len(refs) == 1:
        ops.append("copy")
    return ops or None


def scan_frag_rules(path, proc, lines, findings, exempt_file):
    """Runs frag-walk and tile-op-miss over one device proc body.

    - frag-walk fires per fragment-reference line unless the file sits in a walk-exempt module or the proc carries its own allowlist marker
    - tile-op-miss fires per frag assignment whose right side is single-frag arithmetic inside the tile op surface
    """
    body = proc["body"]
    codes = [c for _no, c in body if c is not None and not c.startswith("EMIT:")]
    allowed = _allowlisted_proc(proc, lines)
    all_codes = [c for _no, c in body]
    for pos, (no, code) in enumerate(body):
        if code is None or code.startswith("EMIT:"):
            continue
        refs = FRAG_REF_RE.findall(code)
        if refs:
            if not exempt_file and not allowed:
                findings.append(Finding(
                    path, no, "frag-walk",
                    "a device proc walks .frags[..][..].frag[..] by hand; "
                    "use the tile_algebra ops, or mark the proc with a "
                    "`# tiles-allow: <what> needs <primitive>` line naming "
                    "the primitive the walk waits for"))
            m = FRAG_LHS_ASSIGN_RE.search(code)
            if m and not exempt_file:
                rhs, _ = _join_rhs(all_codes, pos + 1, code[m.end():])
                refs_all = FRAG_REF_RE.findall(rhs)
                if refs_all and len(_frag_indices(refs_all)) == 1:
                    ops = _classify_single_frag(rhs, refs_all)
                    if ops:
                        findings.append(Finding(
                            path, no, "tile-op-miss",
                            "single-frag arithmetic restates a tile op, use "
                            + " + ".join(ops)
                            + " from tile_algebra (tile_ops_unary/tile_ops_binary)"))


def scan_lane_cell(path, codes, findings):
    """Runs the decomposition inventory over the device proc body lines.

    Contract:

    - the manual crd2idx + mod/div sites report the pending `laneCellOf` primitive
    - the raw `.data[0]` reads report the pending `rowScalar` accessor
    - both name tile_algebra work, not a proc that exists today
    """
    for no, code in codes:
        if code is None or code.startswith("EMIT:"):
            continue
        if CRD2IDX_RE.search(code) or CELL_MOD_DIV_RE.search(code):
            findings.append(Finding(
                path, no, "lane-cell",
                "manual lane-to-cell decomposition (crd2idx + cell mod/div): "
                "a `laneCellOf`-class helper is a pending tile_algebra "
                "primitive, keep this site in the inventory"))
        elif SCALAR_EXTRACT_RE.search(code):
            findings.append(Finding(
                path, no, "scalar-extract",
                "raw .data[0] scalar extraction from a row vector: a "
                "`rowScalar` accessor is a pending tile_algebra primitive, "
                "keep this site in the inventory"))


def scan_magic_dim(path, code, no, findings):
    """Runs magic-dim over one device proc body line."""
    stripped = code.strip()
    if CONST_DEF_RE.match(code):
        return
    m = RT_CALL_RE.search(code)
    if m:
        depth = 1
        j = m.end()
        while j < len(code) and depth:
            if code[j] == "(":
                depth += 1
            elif code[j] == ")":
                depth -= 1
            j += 1
        args = code[m.end():j - 1].split(",")
        for a in args:
            a = a.strip()
            if re.fullmatch(r"(8|16|32|64|128|256)", a):
                findings.append(Finding(
                    path, no, "magic-dim",
                    "a bare literal sets a register-tile dim, bind it to a "
                    "named const or an atom getter (getM/getN)"))
                return
    if GETTER_RE.search(code) and re.search(r"\bdoAssert\b|\bstatic\b", code):
        return
    if (re.search(r"\bdoAssert\b", code) and MAGIC_LIT_RE.search(code)
            and not re.search(r"\bin\s*\{|\s\.\.\s", code)):
        findings.append(Finding(
            path, no, "magic-dim",
            "a bare tile-dim literal in an assertion, name the dim (atom "
            "getter or const) instead of the literal"))
        return
    mm = re.search(r"\b(?:mod|div)\s+(8|16|32|64|128|256)\b", code)
    if mm and re.search(r"\.\.\s*<?\s*%s\b" % mm.group(1), code):
        # a pure `0 ..< L` loop bound carries the literal alone, the named-dim demand targets derived bounds
        mm = None
    if mm:
        findings.append(Finding(
            path, no, "magic-dim",
            "a bare literal drives the lane/cell arithmetic, name the dim "
            "(atom getter or const)"))
        return
    if (re.search(r"\barray\s*\[\s*(8|16|32|64|128|256)\s*,", code)
            and not CONST_DEF_RE.match(code)):
        findings.append(Finding(
            path, no, "magic-dim",
            "a bare literal sizes a per-lane array, name the dim (a const "
            "such as MaxTopK)"))
        return
    if (not FOR_HEADER_RE.match(code)
            and re.search(r"[*+]\s*(8|16|32|64|128|256)\b(?![\d.'])"
                          r"|\b(8|16|32|64|128|256)\s*[*+]", code)
            and not re.search(r"[eE][+-]?\d", code)):
        findings.append(Finding(
            path, no, "magic-dim",
            "a bare literal scales the index arithmetic, name the dim "
            "(atom getter or const)"))
        return


def scan_raw_emit(path, emit_text, no, findings, builtins):
    """Fires when one {.emit:} block spells a Crucible builtin by hand."""
    for name in builtins:
        if re.search(r"\b%s\b" % re.escape(name), emit_text):
            findings.append(Finding(
                path, no, "raw-emit",
                "a {.emit:} block spells %s by hand, call the Crucible "
                "builtin (builtins_catalog)" % name))
            return


FP32_LOWEST = (struct.pack("<f", -3.402823466e38), "fp32Lowest (pending "
               "math_consts)")


def load_math_consts(paths):
    """Parses the shared math_consts values from the scanned tree.

    Returns a list of (float32 bits, name) pairs from every
    `const NAME* = <value>'f32` definition in a file named math_consts.nim
    under the scan paths."""
    consts = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.name == "math_consts.nim":
            files = [pp]
        elif pp.is_dir():
            files = sorted(pp.rglob("math_consts.nim"))
        else:
            continue
        for f in files:
            for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
                m = CONST_FLOAT_RE.match(raw)
                if m:
                    bits = struct.pack("<f", float(m.group(2)))
                    consts.append((bits, m.group(1)))
    consts.append(FP32_LOWEST)
    return consts


def scan_math_const(path, code, no, findings, consts):
    """Fires when a float literal bitwise matches a shared math_consts value."""
    for m in FLOAT_LIT_RE.finditer(code):
        try:
            bits = struct.pack("<f", float(m.group(1)))
        except (ValueError, OverflowError):
            continue
        for cbits, name in consts:
            if bits == cbits:
                findings.append(Finding(
                    path, no, "math-const",
                    "a float literal re-spells %s from math_consts.nim, use "
                    "the named constant" % name))
                return


def scan_doc_shape(path, proc, lines, findings, exempt_file):
    """Runs kernel-doc-shape over one exported device proc.

    Contract:

    - the doc block is the run of `##` lines directly after the `=` terminator
    - the SDPA form carries one of the section markers
      (Parameters:/Expected input/Output:), callers read the shape from the doc
    - bare trailing `#` comments beside the signature never document the shape
    """
    if exempt_file or not (proc["device"] and proc["exported"]):
        return
    eq = None
    for k in range(proc["header"], proc["end"]):
        if PRAGMA_END_EQ_RE.search(lines[k].rstrip()):
            eq = k
            break
    if eq is None:
        return
    doc = []
    j = eq + 1
    while j < len(lines):
        s = lines[j].strip()
        if s.startswith("##"):
            doc.append(s)
            j += 1
        elif s and j > eq + 1:
            break
        else:
            j += 1
    if not doc:
        findings.append(Finding(
            path, proc["start"], "kernel-doc-shape",
            "an exported kernel carries no doc block, open the body with "
            "the SDPA form (## Expected input: / ## Output: bullets, see "
            "k_tile_gemm.gemm_with_bias_epilogue)"))
        return
    bulleted = any("- " in d for d in doc)
    if (not any(marker in d for d in doc for marker in DOC_PARAM_MARKERS)
            and not bulleted):
        findings.append(Finding(
            path, proc["start"], "kernel-doc-shape",
            "the kernel doc block states no input/output shape, add the "
            "SDPA sections (## Expected input: / ## Output:, see "
            "k_tile_gemm.gemm_with_bias_epilogue) instead of trailing `#` "
            "parameter comments"))


def load_builtins():
    """Reads the Crucible builtin names from builtins_catalog.nim.

    Returns the bare proc names declared with {.builtin.}, the raw-emit
    check flags emit blocks spelling these by hand."""
    root = _repo_root_guess()
    catalog = None
    if root:
        cand = Path(root) / "workspace" / "crucible" / "src" / "codegen" \
            / "builtins" / "builtins_catalog.nim"
        if cand.is_file():
            catalog = cand
    if catalog is None:
        return ("threadgroup_barrier",)
    names = []
    for raw in catalog.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*(?:proc|let|template)\s+(\w+)\*?\s*"
                     r"(?:\([^)]*\))?\s*(?::[^=]+)?\s*(?:=|{\.builtin)", raw)
        if m and "{.builtin" in raw:
            names.append(m.group(1))
    return tuple(names) or ("threadgroup_barrier",)


def _repo_root_guess():
    """Returns the working directory when it looks like the repo root,
    None otherwise. The builtins catalog lookup stays best-effort."""
    cwd = Path.cwd()
    if (cwd / "workspace" / "crucible" / "src" / "codegen").is_dir():
        return str(cwd)
    return None


def collect_files(paths):
    """Collects the .nim files under the scan paths, sorted and deduped,
    defaulting to the two kernel roots when no path is given."""
    roots = paths or list(KERNEL_ROOTS)
    files = []
    for p in roots:
        pp = Path(p)
        if not pp.exists():
            sys.stderr.write("lint_tiles: missing path: %s\n" % p)
            sys.exit(2)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("*.nim")))
        elif pp.suffix == ".nim":
            files.append(pp)
    out, seen = [], set()
    for f in sorted(set(files)):
        r = f.resolve()
        if "nimcache" in r.parts or r in seen:
            continue
        seen.add(r)
        out.append(f)
    return out


def decls(lines):
    """Extracts every callable declaration of one file.

    Returns a list of records.

| field      | contents                                                                                |
| ---------- | --------------------------------------------------------------------------------------- |
| name       | the declared identifier, None for a shape the matcher drops (backticked names)          |
| kind       | proc, func, template, macro, iterator, or converter                                     |
| start      | the 1-based header line                                                                 |
| eq_line    | the 1-based line carrying the depth-0 `=` terminator, None for a bodyless declaration   |
| generics   | the declared generic parameter names, [] when the name carries no bracket list          |
| value_text | the signature's parameter text, the first balanced parenthesis group after the generics |
| body       | code lines after the `=` terminator, doc and comment lines excluded                     |

    A signature spans from its header to the first `=` at bracket depth 0,
    so mid-parameter defaults never terminate it.
    """
    out = []
    i, n = 0, len(lines)
    while i < n:
        code, _c = _strip_comment(lines[i])
        if not CALLABLE_HEAD_RE.match(code):
            i += 1
            continue
        nm = CALLABLE_NAME_RE.match(lines[i])
        kind = CALLABLE_HEAD_RE.match(code).group(0).strip()
        name = nm.group(2) if nm else None
        indent = len(lines[i]) - len(lines[i].lstrip())
        depth, eq_line, eq_col = 0, None, None
        j = i
        while j < n and j <= i + 25:
            line = lines[j]
            k = 0
            while k < len(line):
                ch = line[k]
                if ch in "([":
                    depth += 1
                elif ch in ")]":
                    depth = max(0, depth - 1)
                elif ch == "=" and depth == 0 and k > 0 \
                        and line[k - 1] not in "=<>!+-*/%~?":
                    eq_line, eq_col = j + 1, k
                    break
                k += 1
            if eq_line is not None:
                break
            j += 1
        # generic parameter names, the first bracket group after the name
        generics = []
        value_text = ""
        if name is not None and eq_line is not None:
            sig_text = " ".join(_strip_comment(lines[t])[0]
                                for t in range(i, eq_line))
            nm2 = re.search(re.escape(name) + r"\s*\*?\s*\`?\s*"
                            r"(?:\[[^\[\]]*\])?", sig_text)
            seg = sig_text[nm2.start():] if nm2 else ""
            gm = re.search(r"\[([^\[\]]*)\]", seg)
            if gm:
                for piece in re.split(r"[;,]", gm.group(1)):
                    pm = re.match(r"\s*([A-Za-z_]\w*)", piece)
                    if pm:
                        generics.append(pm.group(1))
            pm = re.search(r"\(([^()]*)\)", seg)
            if pm:
                value_text = pm.group(1)
        # the body count reads the code lines between the terminator and the dedent
        body_count = 0
        if eq_line is not None:
            rest = _strip_comment(lines[eq_line - 1][eq_col + 1:])[0]
            if rest.strip():
                body_count += 1
            k = eq_line
            while k < n:
                line = lines[k]
                if not line.strip():
                    k += 1
                    continue
                if len(line) - len(line.lstrip()) <= indent:
                    break
                code, comment_only = _strip_comment(line)
                if code.strip():
                    body_count += 1
                k += 1
        out.append({"name": name, "kind": kind, "start": i + 1,
                    "eq_line": eq_line, "indent": indent,
                    "generics": generics, "value_text": value_text,
                    "body": body_count})
        i = max(i + 1, eq_line or i + 1)
    return out


def scan_hash_above_proc(path, lines, blocked, findings):
    """Runs hash-above-proc and divider-space over one file's raw lines.

    - a bare `#` comment line sitting immediately above a proc, func,
      or template header fires at the comment block's first line
    - a section divider needs a blank line before and after, line 1
      keeps its before-side exempt
    """
    for d in decls(lines):
        if d["name"] is None or d["kind"] not in RULE_KINDS:
            continue
        h = d["start"] - 1
        if h == 0:
            continue
        above = lines[h - 1]
        s = above.strip()
        if not s.startswith("#") or s.startswith("##") or (h) in blocked:
            continue
        if DIVIDER_RE.match(above):
            continue
        k = h - 1
        while k > 0:
            t = lines[k - 1].strip()
            if not t.startswith("#") or t.startswith("##") or k in blocked:
                break
            k -= 1
        findings.append(Finding(
            path, k + 1, "hash-above-proc",
            "a # comment above the %s %s does not attach, use a ## doc "
            "comment; Nim reads ##" % (d["kind"], d["name"])))
    # Spacing reads the whole comment run as one unit, whether a divider,
    # a sandwich banner (rule, title, rule), or a rule with its section
    # prose. Air sits between that block and the code around it.
    i = 0
    while i < len(lines):
        if (i + 1) in blocked or not lines[i].strip().startswith("#")                 or lines[i].strip().startswith("##"):
            i += 1
            continue
        start = i
        while i + 1 < len(lines) and lines[i + 1].strip().startswith("#") \
                and lines[i + 1].strip()[:2] != "##" \
                and (i + 2) not in blocked:
            i += 1
        run = lines[start:i + 1]
        if not any(DIVIDER_RE.match(r) for r in run):
            i += 1
            continue
        first_div = start + next(
            j for j, r in enumerate(run) if DIVIDER_RE.match(r)) + 1
        if start > 0 and lines[start - 1].strip():
            findings.append(Finding(
                path, first_div, "divider-space",
                "a section divider keeps a blank line before and after it"))
        if i + 1 < len(lines) and lines[i + 1].strip():
            findings.append(Finding(
                path, first_div, "divider-space",
                "a section divider keeps a blank line before and after it"))
        i += 1


def scan_one_liner(path, ds, findings):
    """Runs the one-liner advisory over a file's callable declarations.

    Contract:
    - the advisory names the wrapper, it never blocks
    - a single-line body is a shape the review reads, not a defeat
    """
    for d in ds:
        if d["name"] is None or d["kind"] not in RULE_KINDS:
            continue
        if d["eq_line"] is not None and d["body"] == 1:
            findings.append(Finding(
                path, d["start"], "one-liner",
                "the body of %s %s is a single line, name the wrapper shape "
                "in the doc so the entry stays evident at review"
                % (d["kind"], d["name"]), warning=True))


def build_generic_map(files):
    """Parses the generic signatures of every scanned file's declarations.

    Returns the {name, path} keyed record map that the explicit-generics
    rule consults, one record per declaration carrying a `=` terminator.

    Record fields:

    - generic names
    - parameter text
    - kind
    """
    out = {}
    for f, lines in files:
        for d in decls(lines):
            if (d["name"] is None or d["kind"] not in RULE_KINDS
                    or d["eq_line"] is None or not d["generics"]):
                continue
            out[(d["name"], f)] = (d["generics"], d["value_text"], d["kind"])
    return out


def scan_explicit_generics(path, lines, ds, generic_map, findings):
    """Runs the explicit-generics rule over one file's call sites.

    Fires when a call site spells a bracketed generic argument list while
    the callee's generic parameters are all bound by its value parameters,
    the compiler derives those from the argument types alone.

    Quiet cases:

    - the callee has no parsed declaration in the scan set
      (type constructors, foreign templates, anything unresolvable)
    - the call sits in the callee's own module, the definition-side entry
      over a core (dense_linear_tile_core[El, 64]) stays legal
    - the bracket count disagrees with the generic parameter count, one
      generic parameter unbound by any value parameter, a tile-width
      literal the compiler cannot derive
    """
    header_lines = {(d["name"], d["start"]) for d in ds if d["name"]}
    by_name = {}
    for (name, f), rec in generic_map.items():
        by_name.setdefault(name, []).append((f, rec))
    for i, raw in enumerate(lines):
        code, _c = _strip_comment(raw)
        for m in CALL_SITE_RE.finditer(code):
            callee, args_text = m.group(1), m.group(2)
            if (callee, i + 1) in header_lines:
                continue
            records = by_name.get(callee)
            if not records:
                continue
            if any(f == path for f, _rec in records):
                continue
            args = [a.strip() for a in args_text.split(",")]
            if not all(len(args) == len(gen) and gen
                       and all(re.search(r"\b%s\b" % re.escape(g), vt)
                               for g in gen)
                       for _f, (gen, vt, _k) in records):
                continue
            findings.append(Finding(
                path, i + 1, "explicit-generics",
                "the call %s[%s] spells generic arguments the compiler "
                "infers from the value arguments, drop the bracket list"
                % (callee, ", ".join(args))))


def scan(path, text, findings, consts, builtins, generic_map=None):
    """Runs every rule over one file, appending to findings.

    Contract:
    - `generic_map` carries the callee signatures parsed across the whole
      scan set, lint() builds it
    - None makes the rule see this file alone, the same-module exemption
      stays exact in single-file use
    """
    lines = text.splitlines()
    procs = device_procs(lines)
    exempt_file = _walk_exempt(path)
    lane_exempt = _lane_cell_exempt(path)
    blocked = nim_block_comment_lines(text)
    ds = decls(lines)
    if generic_map is None:
        generic_map = build_generic_map([(path, lines)])
    scan_hash_above_proc(path, lines, blocked, findings)
    scan_one_liner(path, ds, findings)
    scan_explicit_generics(path, lines, ds, generic_map, findings)
    for proc in procs:
        if not proc["device"]:
            continue
        scan_frag_rules(path, proc, lines, findings, exempt_file)
        scan_doc_shape(path, proc, lines, findings, exempt_file)
        codes = [(no, c) for no, c in proc["body"]]
        if not lane_exempt:
            scan_lane_cell(path, codes, findings)
        for no, code in codes:
            if code is None:
                continue
            if code.startswith("EMIT:"):
                scan_raw_emit(path, code[len("EMIT:"):], no, findings,
                              builtins)
                continue
            scan_magic_dim(path, code, no, findings)
            scan_math_const(path, code, no, findings, consts)


def lint(paths, base=None):
    """Returns the sorted findings for every collected file under paths.

    With base set, findings are scoped to the lines the diff from that commit adds, the added-line scoping lint_docs applies."""
    files = collect_files(paths)
    consts = load_math_consts(files)
    builtins = load_builtins()
    texts = [(f, f.read_text(encoding="utf-8", errors="replace")) for f in files]
    generic_map = build_generic_map(
        [(f, t.splitlines()) for f, t in texts])
    findings = []
    cache = {}
    for f, text in texts:
        file_findings = []
        scan(f, text, file_findings, consts, builtins, generic_map)
        if base:
            added = added_lines(f, base, cache)
            if added is None:
                findings.extend(file_findings)
            else:
                findings.extend(fd for fd in file_findings if fd.line in added)
        else:
            findings.extend(file_findings)
    findings.sort(key=lambda x: (str(x.path), x.line, x.rule))
    return findings


def main(argv):
    """CLI entry point in the lint_docs exit-code shape.

    - exit 0 clean
    - exit 1 counted findings
    - exit 2 usage or tool failure
    """
    args = argv[1:]
    base = None
    if "--base" in args:
        i = args.index("--base")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        base = args[i + 1]
        args = args[:i] + args[i + 2:]
    base = base or os.environ.get("TILE_LINT_BASE") or None
    stats = "--stats" in args
    args = [a for a in args if a != "--stats"]
    paths = args or list(KERNEL_ROOTS)
    findings = lint(paths, base)
    for fd in findings:
        print("%s:%d: %s: %s" % (fd.path, fd.line, fd.rule, fd.reason))
    counts = Counter(fd.rule for fd in findings)
    print("%d findings: %s" % (
        len(findings),
        ", ".join("%s %d" % (r, counts[r]) for r in sorted(counts))))
    if stats:
        per_file = Counter(str(fd.path) for fd in findings)
        for p, c in sorted(per_file.items()):
            print("  %s: %d" % (p, c))
    return 1 if any(not fd.warning for fd in findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
