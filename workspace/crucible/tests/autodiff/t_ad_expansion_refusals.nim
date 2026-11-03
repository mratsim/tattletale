# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## bwd_diff expansion refusals of the backward derivation and the pair expansion
##
## Test that every refusal case refuses at compile time with the `bwd_diff:` prefix and the named construct
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/autodiff/t_ad_expansion_refusals.nim

import std/[os, osproc, strutils]

proc repoRoot(): string =
  ## This file lives five levels below the repository root.
  result = currentSourcePath()
  for i in 0 ..< 5:
    result = result.parentDir()

proc runCase(name, src: string): tuple[output: string, exitCode: int] =
  ## Compiles one standalone snippet with `nim check` and returns the combined
  ## compiler output. A shared nimcache keeps the crucible sources compiled
  ## once across cases.
  let dir = repoRoot() / "build" / "ad_expansion_refusals"
  createDir(dir)
  let path = dir / ("case_" & name & ".nim")
  writeFile(path, src)
  let cmd = "nim check --hints:off --warnings:off --verbosity:0 " &
            "--maxLoopIterationsVM:100000000 --nimcache:build/nimcache_ad_expansion_refusals " &
            "--path:. " & path
  result = execCmdEx(cmd, options = {poStdErrToStdOut, poUsePath},
                     workingDir = repoRoot())

var failures: seq[string]
var caseCount = 0

proc expectRefusal(name, src, construct: string) =
  inc caseCount
  let (output, code) = runCase(name, src)
  if code == 0:
    failures.add "case '" & name & "' must refuse to compile, it compiled cleanly"
    return
  if "bwd_diff:" notin output:
    failures.add "case '" & name & "' refusal must carry the bwd_diff: prefix, got:\n" & output
    return
  if construct notin output:
    failures.add "case '" & name & "' refusal must name the construct '" & construct &
      "', got:\n" & output

proc entryBody(params, body: string, pragmas = "global", backend = "cgMetal", prelude = "", tail = ""): string =
  ## Standalone snippet driving the `bwd_diff` macro entry over one kernel:
  ## the kernel typechecks at its definition, then the pair expansion runs
  ## the full ingestion pipeline.
  result = "import workspace/crucible\n"
  if prelude.len > 0:
    result.add prelude & "\n"
  result.add "proc kernel(" & params & ") {." & pragmas & ".} =\n" & body & "\n"
  if tail.len > 0:
    result.add tail & "\n"
  result.add "const adKernels = " & backend & ".bwd_diff(kernel)\n"

proc declaredBody(fwdBody, bwdProc, bwdArg = "kernel_bwd", tail = ""): string =
  ## Standalone snippet of a declared pair, `bwd_diff(kernel, <bwdArg>)`:
  ## one forward kernel and one hand-written backward.
  result = "import workspace/crucible\n\n"
  result.add "proc kernel(dst, a: ptr UncheckedArray[float32]) {.global.} =\n"
  result.add fwdBody & "\n"
  result.add bwdProc & "\n"
  result.add "const adKernels = cgMetal.bwd_diff(kernel, " & bwdArg & ")\n"
  if tail.len > 0:
    result.add tail & "\n"

const BwdNameRow = "must be named 'kernel_bwd', the pair's launch name"
const MirrorRow = "does not mirror"
const ExclusiveRow = "the loop condition must be an exclusive < comparison over the loop variable"
const RetRow = "mid-kernel returns have no reverse control flow"

# ── Derivation refusals: the loop idiom ──────────────────────────────────

expectRefusal("inclusive_bound", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  for i in 0 .. 8:\n    dst[i] = a[i] * 2.0'f32"), ExclusiveRow)

expectRefusal("countup_step_two", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  for i in countup(0, 7, 2):\n    dst[i] = a[i] * 2.0'f32"), ExclusiveRow)

expectRefusal("sibling_loops", entryBody(
  "dst, a, b: ptr UncheckedArray[float32], n1, n2: int32, w: float32",
  """  for i in 0 ..< n1:
    dst[i] = a[i]
  for j in 0 ..< n2:
    dst[j] = b[j] * w"""), "more than one top-level loop")

expectRefusal("nested_loops", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  for i in 0 ..< 4:
    for j in 0 ..< 4:
      dst[j] = a[i] * 2.0'f32"""), "nested loops are outside the differentiable subset")

expectRefusal("raw_while", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  var i = 0
  while i < 8:
    dst[i] = a[i] * 2.0'f32
    i += 1"""), "a loop without the desugared-for preamble is data-dependent")

expectRefusal("binop_bound", entryBody(
  "dst: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], alpha: float32, n: int32",
  "  for i in 0 ..< (2 * n):\n    dst[i] = alpha * a[i]"),
  "the loop bound must be an int literal or a kernel parameter")

# ── Derivation refusals: returns, locals, slots ──────────────────────────

expectRefusal("branch_return", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  for i in 0 ..< 8:
    if a[i] > 0.0'f32:
      return
    dst[i] = a[i] * 2.0'f32"""), RetRow)

expectRefusal("mid_position_return", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  dst[0] = a[0]
  return
  dst[1] = a[1]"""), RetRow)

expectRefusal("uninitialized_local", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  var x: float32
  for i in 0 ..< 8:
    x = a[i] * 2.0'f32
    dst[i] = x"""), "declares no initializer")

expectRefusal("non_scalar_local", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  let tmp = [a[0], a[1]]
  for i in 0 ..< 8:
    dst[i] = a[i] * 2.0'f32"""), "locals are scalars")

expectRefusal("local_redefined", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  var x = a[0] * 2.0'f32
  x = a[1]
  for i in 0 ..< 8:
    dst[i] = x"""), "assignment redefines local 'x', loop-carried mutation is outside the subset")

expectRefusal("slot_written_twice", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  for i in 0 ..< 8:
    dst[i] = a[i]
    dst[i] = a[i] * 2.0'f32"""), "buffer slot 'dst@i' is written more than once")

expectRefusal("in_place_update", entryBody(
  "dst: ptr UncheckedArray[float32]",
  "  for i in 0 ..< 8:\n    dst[i] = dst[i] * 2.0'f32"),
  "reads buffer 'dst' it also writes")

expectRefusal("no_buffer_write", entryBody(
  "a: ptr UncheckedArray[float32]",
  "  for i in 0 ..< 8:\n    discard a[i]"),
  "writes no buffer, nothing to differentiate")

expectRefusal("scalar_gradient_outside_loop", entryBody(
  "dst, a: ptr UncheckedArray[float32], alpha: float32",
  "  dst[0] = alpha * a[0]"), "arises outside the index loop")

# ── Derivation refusals: calls, conversions, injection ───────────────────

expectRefusal("atomic_scalar_target", entryBody(
  "dst, a: ptr UncheckedArray[float32], alpha: float32",
  "  for i in 0 ..< 8:\n    dst[i] = a[i]\n    discard atomicFetchAdd(addr alpha, a[i])"),
  "differentiates over a buffer-slot target")

expectRefusal("statement_call", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  log2(a[0])\n  for i in 0 ..< 8:\n    dst[i] = a[i]",
  prelude = "proc log2(x: float32) = discard"),
  "as a statement is outside the differentiable subset")

expectRefusal("array_literal", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  for i in 0 ..< 8:\n    dst[i] = [a[i], 1.0'f32][0]"),
  "aggregate constructions have no member gradients")

expectRefusal("atomic_value_used", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  for i in 0 ..< 8:
    let old = atomicFetchAdd(addr dst[0], a[i])
    dst[i] = old"""),
  "atomic builtin 'atomicFetchAdd' differentiates only as a statement")

expectRefusal("user_call", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  for i in 0 ..< 8:\n    dst[i] = twice(a[i])",
  prelude = "proc twice(x: float32): float32 = 2.0'f32 * x"),
  "outside the differentiable subset; inline the callee")

expectRefusal("frontend_gated_builtin", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  for i in 0 ..< 8:\n    dst[i] = sin(a[i])",
  prelude = "import std/math"),
  "call to sin has no body to lower")

expectRefusal("bitcast", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  for i in 0 ..< 8:
    let bits = cast[int32](a[i])
    discard bits
    dst[i] = a[i] * 2.0'f32"""),
  "bitcasts have no gradient")

expectRefusal("emit_pragma", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  {.emit: "dst[0] = 1.0f;".}
  for i in 0 ..< 8:
    dst[i] = a[i] * 2.0'f32"""),
  "raw target-source injection is outside the differentiable subset")

expectRefusal("not_a_kernel", """import workspace/crucible

proc twice(x: float32): float32 = 2.0'f32 * x

const adKernels = cgMetal.bwd_diff(twice)
""", "no {.global.} kernel to differentiate")

# ── Expansion refusals: the thread-local write analysis ──────────────────

expectRefusal("forward_race_names_geometry", entryBody(
  "dst, a: ptr UncheckedArray[float32], b: ptr UncheckedArray[float32], alpha: float32",
  "  for i in 0 ..< 8:\n    dst[i] = alpha * a[i] + b[i]",
  pragmas = "global, workgroup: (4, 4)"),
  "at workgroup (4, 4, 1): the forward writes '(pat dst i)', the write index does not separate the threads")

expectRefusal("backward_race_names_geometry", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  let j = thread_position_in_threadgroup.x
  dst[j] = a[0] * a[0]""",
  pragmas = "global, workgroup: (4, 1)"),
  "at workgroup (4, 1, 1): the backward writes '(pat d_a 0)', the write index does not separate the threads")

# ── Expansion refusals: the declared pair ────────────────────────────────

const DeclaredBwdOk = """
proc kernel_bwd(d_dst, d_a, dst, a: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    d_a[i] += d_dst[i] * 2.0'f32
"""

const MyBwd = """
proc myBwd(d_dst, d_a, dst, a: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    d_a[i] += d_dst[i] * 2.0'f32
"""

const MisnamedBwd = """
proc kernel_bwd(d_x, d_a, dst, a: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    d_a[i] += d_x[i] * 2.0'f32
"""

expectRefusal("declared_name", declaredBody("  for i in 0 ..< 8:\n    dst[i] = a[i] * 2.0'f32", MyBwd, bwdArg = "myBwd"), BwdNameRow)
expectRefusal("declared_positional", declaredBody("  for i in 0 ..< 8:\n    dst[i] = a[i] * 2.0'f32", MisnamedBwd), MirrorRow)
const ShortBwd = """
proc kernel_bwd(d_dst, d_a: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    d_a[i] += d_dst[i] * 2.0'f32
"""

const LongBwd = """
proc kernel_bwd(d_dst, d_a, dst, a, d_extra: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    d_a[i] += d_dst[i] * 2.0'f32
"""

expectRefusal("declared_mirror_short", declaredBody("  for i in 0 ..< 8:\n    dst[i] = a[i] * 2.0'f32", ShortBwd),
  "stops at parameter")
expectRefusal("declared_mirror_long", declaredBody("  for i in 0 ..< 8:\n    dst[i] = a[i] * 2.0'f32", LongBwd),
  "beyond the mirror")


# Save and unexpected-argument cases patch the bwd_diff call line of an otherwise valid declared pair

for saveCase in [
  ("save_unknown_name", ", save = [nosuch]", "save name 'nosuch' is not a parameter of 'kernel'"),
  ("save_not_output", ", save = [a]", "save name 'a' is not a forward output of 'kernel'"),
  ("save_not_list", ", save = 3", "save takes a list of forward-output names"),
  ("save_not_name", ", save = [3]", "save takes forward-output names"),
  ("unexpected_argument", ", foo = 3", "only save = [<forward outputs>] is supported"),
]:
  let (name, arg, construct) = saveCase
  let snippet = declaredBody("  for i in 0 ..< 8:\n    dst[i] = a[i] * 2.0'f32", DeclaredBwdOk)
    .replace("const adKernels = cgMetal.bwd_diff(kernel, kernel_bwd)",
             "const adKernels = cgMetal.bwd_diff(kernel, kernel_bwd" & arg & ")")
  expectRefusal(name, snippet, construct)

# ── Expansion refusals: the backend gates ────────────────────────────────

expectRefusal("backend_gate", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  "  for i in 0 ..< 8:\n    dst[i] = a[i] * 2.0'f32",
  backend = "cgCuda"),
  "backend kind cgCuda has no autodiff pipeline, this op gates Metal and Vulkan GLSL only")

expectRefusal("coord_forward_race", entryBody(
  "dst, a: ptr UncheckedArray[float32]",
  """  let gx = thread_position_in_threadgroup.x
  for i in 0 ..< 8:
    dst[i] = a[gx + i] * 2.0'f32""",
  pragmas = "global, workgroup: (8, 1)"),
  "at workgroup (8, 1, 1): the forward writes '(pat dst i)', the write index does not separate the threads")

# setupAutoDiff refuses a non-AD engine kind at compile time, without the bwd_diff prefix:
# pair compilation succeeded, so this refusal names the engine kind
block:
  inc caseCount
  let (output, code) = runCase("setup_gate", """import workspace/crucible

proc kernel(dst, a: ptr UncheckedArray[float32]) {.global.} =
  for i in 0 ..< 8:
    dst[i] = a[i] * 2.0'f32

const adKernels = cgMetal.bwd_diff(kernel)

proc runTest() =
  var engine = newEngine(runCuda)
  let ad = engine.setupAutoDiff(adKernels)

runTest()
""")
  doAssert code != 0, "case 'setup_gate' must refuse to compile, it compiled cleanly"
  doAssert "setupAutoDiff: this engine has no autodiff pipeline, this op gates Metal and Vulkan GLSL" in output,
    "case 'setup_gate' refusal must name the engine gate, got:\n" & output

if failures.len > 0:
  for f in failures:
    echo "[FAIL] ", f
  echo "[FAIL] bwd_diff expansion refusals: ", failures.len, " failing case(s)"
  quit(1)
echo "[OK] bwd_diff expansion refusals: ", caseCount, " cases"
