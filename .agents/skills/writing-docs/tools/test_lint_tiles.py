#!/usr/bin/env python3
"""Self-test for lint_tiles.py.

    $ python3 .agents/skills/writing-docs/tools/test_lint_tiles.py

Cases:

- one synthetic .nim source per rule class
- the module-scope exemption and the proc-scope allowlist marker

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
             [(L3, "frag-walk"), (L3, "tile-op-miss")])
ok &= expect("scale suggestion",
             run(PROC % "  d.frags[n][m].frag[v] = s.frags[n][m].frag[v] * scale\n"),
             [(L3, "frag-walk"), (L3, "tile-op-miss")])
ok &= expect("gather from memory is not a tile op",
             run(PROC % "  d.frags[n][m].frag[v] = w[rowByte + 8 + sb]\n"),
             [(L3, "frag-walk"), (L3, "magic-dim")])

# the hand-inlined crd2idx decomposition reports the pending primitive,
# the raw .data[0] row-vector read reports rowScalar
ok &= expect("lane-cell fires",
             run(PROC % "  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()\n"),
             [(L3, "lane-cell")])
ok &= expect("scalar-extract fires",
             run(PROC % "  let g = gVec.data[0]\n"), [(L3, "scalar-extract")])

# the rt dim literal and the mod/div literal fire, the named dim and the pure loop bound stay quiet
ok &= expect("rt dim literal",
             run(PROC % "  var t: rt_l(float32, 8, Dk)\n"), [(L3, "magic-dim")])
ok &= expect("mod literal",
             run(PROC % "  let row = cell mod 8\n"),
             [(L3, "magic-dim"), (L3, "lane-cell")])
ok &= expect("named rt dim quiet",
             run(PROC % "  var t: rt_l(float32, TileR, Dk)\n"), [])
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
                 consts=consts), [(L3, "math-const")])
ok &= expect("math-const named const quiet",
             run(PROC % "  let s = x * Log2e\n", consts=consts), [])

# no doc block and no shape sections fire, the SDPA form and the bulleted
# shape doc stay quiet
PROC_NODOC = """proc k*[T](x: ptr UncheckedArray[T]) {.device.} =
%s
"""
ok &= expect("kernel-doc-shape, no doc block", run(PROC_NODOC % "  discard\n"),
             [(1, "kernel-doc-shape")])
sdpa = ("  ## Expected input:\n"
        "  ##   - x, shape (M, C)\n"
        "  ##\n"
        "  ## Output:\n"
        "  ##   - the normed rows\n")
ok &= expect("SDPA doc quiet", run(PROC % sdpa), [])
ok &= expect("one-line formula doc fires",
             run(PROC_NODOC % "  ## d = a + b\n  discard\n"),
             [(1, "kernel-doc-shape")])

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

cases = 16
print("ALL PASS" if ok else "FAILURES", "(%d cases)" % cases)
sys.exit(0 if ok else 1)
