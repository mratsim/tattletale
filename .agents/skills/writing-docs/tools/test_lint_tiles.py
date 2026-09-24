#!/usr/bin/env python3
"""Self-test for lint_tiles.py.

    $ python3 .agents/skills/writing-docs/tools/test_lint_tiles.py

Cases:

- one synthetic .nim source per rule class
- the module-scope exemption and the proc-scope allowlist marker
- the structural rules, hash-above-proc, divider-space, the one-liner advisory, the explicit-generics call-site ban

Each case builds a device-proc snippet in memory, runs scan(), asserts the expected rule ids at the expected lines.
"""
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint_tiles as lt

# One exported {.device.} kernel, the body spliced in at %s.
PROC = """proc k*[T](x: ptr UncheckedArray[T]) {.device.} =
  ## Expected input:
  ##   - x, one row per lane
  ##
  ## Output:
  ##   - the elementwise map over x
%s
"""


def run(snippet, rel="workspace/positron/src/kernels/ceramic/x.nim",
        consts=(), builtins=("threadgroup_barrier",)):
    """Scans one synthetic snippet through lint_tiles.scan.

    Returns the (line, rule) pairs of the findings.
    """
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(snippet)
        findings = []
        lt.scan(f, snippet, findings, list(consts), builtins)
        return [(x.line, x.rule) for x in findings]


def expect(label, got, want):
    """Compares the sorted finding pairs, printing PASS or FAIL per case."""
    ok = sorted(got) == sorted(want)
    print(("PASS" if ok else "FAIL"), label,
          "" if ok else "got %s want %s" % (got, want))
    return ok


# the snippet body starts at line 7, the proc header plus its 5-line doc
# block precede it, so each case's first body line is line 7.
L3 = 7

ok = True
# frag-walk fires on the nested walk, the tile_algebra module silences it,
# the allowlist marker scopes to its own proc
body = ("  for n in 0 ..< 2:\n"
        "    for m in 0 ..< 2:\n"
        "      for v in 0 ..< 2:\n"
        "        d.frags[n][m].frag[v] = s.frags[n][m].frag[v]\n")
ok &= expect("frag-walk fires", run(PROC % body),
             [(L3 + 3, "frag-walk"), (L3 + 3, "tile-op-miss")])
ok &= expect("frag-walk exempt in tile_algebra",
             run(PROC % body,
                 rel="workspace/ceramic/src/tile_algebra/x.nim"), [])
marked = ("  # tiles-allow butterflyCore needs a subgroup butterfly "
          "primitive\n") + body
ok &= expect("allowlist marker scopes to its proc",
             run(PROC % marked), [(L3 + 4, "tile-op-miss")])

# the copy and the scale spell existing ops, the gather from memory is
# not on the op surface
ok &= expect("copy suggestion",
             run(PROC % "  d.frags[n][m].frag[v] = s.frags[n][m].frag[v]\n"),
             [(1, "one-liner"), (L3, "frag-walk"), (L3, "tile-op-miss")])
ok &= expect("scale suggestion",
             run(PROC % "  d.frags[n][m].frag[v] = s.frags[n][m].frag[v] * scale\n"),
             [(1, "one-liner"), (L3, "frag-walk"), (L3, "tile-op-miss")])
ok &= expect("gather from memory is not a tile op",
             run(PROC % "  d.frags[n][m].frag[v] = w[rowByte + 8 + sb]\n"),
             [(1, "one-liner"), (L3, "frag-walk"), (L3, "magic-dim")])

# the hand-inlined crd2idx decomposition reports the pending primitive,
# the raw .data[0] row-vector read reports rowScalar
ok &= expect("lane-cell fires",
             run(PROC % "  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()\n"),
             [(1, "one-liner"), (L3, "lane-cell")])
ok &= expect("scalar-extract fires",
             run(PROC % "  let g = gVec.data[0]\n"),
             [(1, "one-liner"), (L3, "scalar-extract")])

# the rt dim literal and the mod/div literal fire, the named dim and the pure loop bound stay quiet
ok &= expect("rt dim literal",
             run(PROC % "  var t: rt_l(float32, 8, Dk)\n"),
             [(1, "one-liner"), (L3, "magic-dim")])
ok &= expect("mod literal",
             run(PROC % "  let row = cell mod 8\n"),
             [(1, "one-liner"), (L3, "magic-dim"), (L3, "lane-cell")])
ok &= expect("named rt dim quiet",
             run(PROC % "  var t: rt_l(float32, TileR, Dk)\n"),
             [(1, "one-liner")])
ok &= expect("pure loop bound quiet",
             run(PROC % "  for i in 0 ..< rowTiles:\n    discard\n"), [])

# the hand-spelled barrier names the Crucible builtin
ok &= expect("raw-emit fires",
             run(PROC % '  {.emit: """\n  threadgroup_barrier(mem_flags::mem_device);\n  """.}\n'),
             [(L3 + 2, "raw-emit")])

# the literal matches the shared table value bitwise, the named constant stays quiet
consts = [(struct.pack("<f", 1.4426950408889634), "Log2e")]
ok &= expect("math-const fires (table)",
             run(PROC % "  let s = x * 1.4426950408889634'f32\n",
                 consts=consts), [(1, "one-liner"), (L3, "math-const")])
ok &= expect("math-const named const quiet",
             run(PROC % "  let s = x * Log2e\n", consts=consts),
             [(1, "one-liner")])

# no doc block and no shape sections fire, the SDPA form and the bulleted
# shape doc stay quiet
PROC_NODOC = """proc k*[T](x: ptr UncheckedArray[T]) {.device.} =
%s
"""
ok &= expect("kernel-doc-shape, no doc block", run(PROC_NODOC % "  discard\n"),
             [(1, "one-liner"), (1, "kernel-doc-shape")])
sdpa = ("  ## Expected input:\n"
        "  ##   - x, shape (M, C)\n"
        "  ##\n"
        "  ## Output:\n"
        "  ##   - the normed rows\n")
ok &= expect("SDPA doc quiet", run(PROC % sdpa), [])
ok &= expect("one-line formula doc fires",
             run(PROC_NODOC % "  ## d = a + b\n  discard\n"),
             [(1, "one-liner"), (1, "kernel-doc-shape")])

# non-device procs stay out of scope
code = """proc host(x: int): int =
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  result = x
"""
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "workspace/positron/src/kernels/ceramic/y.nim"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(code)
    findings = []
    lt.scan(f, code, findings, [], ())
ok &= expect("host proc out of scope", [(x.line, x.rule) for x in findings], [])

# ─── the structural rules ────────────────────────────────────────────

# a # comment immediately above the header fires at the comment block's
# first line, a ## block above or a blank line between stays quiet
NOTE_ABOVE = "# a maintainer note\n" + PROC
ok &= expect("hash-above-proc fires at the block start",
             [r for r in run(NOTE_ABOVE % body) if r[1] == "hash-above-proc"],
             [(1, "hash-above-proc")])
ok &= expect("## above the header quiet",
             [r for r in run("## a doc block above\n" + PROC % body)
              if r[1] in ("hash-above-proc", "divider-space")], [])
ok &= expect("blank line between comment and header quiet",
             [r for r in run("# a maintainer note\n\n" + PROC % body)
              if r[1] == "hash-above-proc"], [])
ok &= expect("divider directly above the header quiet for hash-above-proc",
             [r for r in run("# ─── the section ───\n" + PROC % body)
              if r[1] == "hash-above-proc"], [])

# the divider keeps air on both sides, the sandwich banner counts as one unit
# a divider on line 1 has no before-side and stays exempt
SANDWICH = ("# ═══════════════════════════\n"
            "#  the section\n"
            "# ═══════════════════════════\n\n") + PROC
ok &= expect("sandwich banner with air quiet",
             [r for r in run(SANDWICH % body) if r[1] == "divider-space"], [])
ok &= expect("divider glued to the header fires",
             [r for r in run("# ─── the section ───\n" + PROC % body)
              if r[1] == "divider-space"],
             [(1, "divider-space")])
ok &= expect("divider run without air after fires",
             [r for r in run("# ─── the section ───\n# prose below\n" + PROC % body)
              if r[1] == "divider-space"],
             [(1, "divider-space")])
ok &= expect("divider without air before fires",
             [r for r in run("import workspace/crucible\n"
                             "# ─── the section ───\n\n" + PROC % body)
              if r[1] == "divider-space"],
             [(2, "divider-space")])
ok &= expect("divider on line 1 quiet",
             [r for r in run("# ─── the section ───\n\n" + PROC % body)
              if r[1] == "divider-space"], [])

# the one-liner advisory names the wrapper and never blocks alone
ok &= expect("one-liner advisory on a template wrapper",
             run("template w*(x: int): int =\n  x + 1\n"),
             [(1, "one-liner")])
ok &= expect("multi-line body quiet",
             [r for r in run("template w*(x: int): int =\n  let y = x\n  y + 1\n")
              if r[1] == "one-liner"], [])


def run_pair(a, b):
    """Lints two modules as one scan set through lint, the cross-module
    callee map the explicit-generics rule reads."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "workspace" / "ceramic" / "src"
        root.mkdir(parents=True)
        (root / "mod_a.nim").write_text(a)
        (root / "mod_b.nim").write_text(b)
        findings = lt.lint([str(root)])
        return [(x.path.name, x.line, x.rule, x.warning) for x in findings]


MOD_A = ("proc core*[El](p: ptr UncheckedArray[El], n: int32) {.device.} =\n"
         "  discard p[0]\n")
MOD_B = ("proc user[El](p: ptr UncheckedArray[El]) {.device.} =\n"
         "  core[El](p, 4'i32)\n")
got = [x[:3] for x in run_pair(MOD_A, MOD_B) if x[2] == "explicit-generics"]
ok &= expect("explicit-generics fires cross-module", got,
             [("mod_b.nim", 2, "explicit-generics")])

MOD_A2 = ("proc core2*[El; TileC: static int](p: ptr UncheckedArray[El],"
          " n: int32) {.device.} =\n  discard p[0]\n")
MOD_B2 = ("proc user2[El](p: ptr UncheckedArray[El]) {.device.} =\n"
          "  core2[El, 64](p, 4'i32)\n")
ok &= expect("partly derivable generics quiet",
             [x[:3] for x in run_pair(MOD_A2, MOD_B2)
              if x[2] == "explicit-generics"], [])

MOD_B3 = ("proc core3*[El](p: ptr UncheckedArray[El], n: int32) {.device.} =\n"
          "  discard p[0]\n\n"
          "proc user3[El](p: ptr UncheckedArray[El]) {.device.} =\n"
          "  core3[El](p, 4'i32)\n")
ok &= expect("same-module call site quiet",
             [x[:3] for x in run_pair(MOD_B3, "")
              if x[2] == "explicit-generics"], [])

MOD_B4 = ("proc user4[El](p: ptr UncheckedArray[El]) {.device.} =\n"
          "  let m = SimdgroupMatrix[El, false]()\n  discard p[0]\n")
ok &= expect("unresolved callee quiet",
             [x[:3] for x in run_pair(MOD_B4, "")
              if x[2] == "explicit-generics"], [])

# the advisory alone keeps rc 0, the counted findings keep rc 1
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "workspace/ceramic/src/solo.nim"
    f.parent.mkdir(parents=True)
    f.write_text("proc solo(x: int): int =\n  x + 1\n")
    res = lt.lint([str(Path(d) / "workspace/ceramic/src")])
ok &= expect("one-liner alone never blocks",
             [(x.rule, x.warning) for x in res],
             [("one-liner", True)])

cases = 31
print("ALL PASS" if ok else "FAILURES", "(%d cases)" % cases)
sys.exit(0 if ok else 1)
