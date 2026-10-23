# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## bwd_diff entry refusal suite
##
## Test that every construct outside the differentiable subset refuses at the `bwd_diff` macro expansion.
## The diagnostic carries the `bwd_diff:` prefix and names the construct:
##
## - one auto-runnable case per refused node-kind row reachable through the entry, the `gpuObjConstr` row stays IR-level
## - the structural contracts: compound assignment, loop-carried overwrites, out-of-place reads, nested loops, scalar cotangents
## - the entry contracts: a non-proc symbol and an unsupported backend kind refuse with the same prefix
## - each case compiles a standalone snippet through `nim check` and must fail with the prefix and the construct named
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/autodiff --nimcache:nimcache/tests/autodiff \
##     workspace/crucible/tests/codegen/autodiff/t_bwd_diff_entry_refusals.nim

import std/[os, osproc, strutils]

proc repoRoot(): string =
  ## This file lives six levels below the repository root.
  result = currentSourcePath()
  for i in 0 ..< 6:
    result = result.parentDir()

proc runCase(name, src: string): tuple[output: string, exitCode: int] =
  ## Compiles one standalone snippet with `nim check` and returns the combined
  ## compiler output. A shared nimcache keeps the crucible sources compiled
  ## once across cases.
  let dir = repoRoot() / "build" / "bwd_entry_refusals"
  createDir(dir)
  let path = dir / ("case_" & name & ".nim")
  writeFile(path, src)
  let cmd = "nim check --hints:off --warnings:off --verbosity:0 " &
            "--maxLoopIterationsVM:100000000 --nimcache:nimcache/bwdentryrefusals " &
            "--path:. " & path
  result = execCmdEx(cmd, options = {poStdErrToStdOut, poUsePath},
                     workingDir = repoRoot())

proc expectRefusal(name, src, construct: string) =
  let (output, code) = runCase(name, src)
  doAssert code != 0,
    "case '" & name & "' must refuse to compile, it compiled cleanly"
  doAssert "bwd_diff:" in output,
    "case '" & name & "' refusal must carry the bwd_diff: prefix, got:\n" & output
  doAssert construct in output,
    "case '" & name & "' refusal must name the construct '" & construct &
    "', got:\n" & output

proc entryBody(params, body: string): string =
  ## A standalone snippet driving the `bwd_diff` macro entry over one kernel.
  ## The kernel typechecks at its definition, the pair expansion runs the full ingestion pipeline.
  result = """
import workspace/crucible

proc kernel(""" & params & """) {.global.} =
""" & body & """

const pair = bkGlsl.bwd_diff(kernel)
"""

# ── Refused node kinds (one case per reachable refused row) ───────────────

expectRefusal("while", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  var j = 0.0'f32
  while j < 4.0'f32:
    j = j + 1.0'f32"""), "gpuWhile")

expectRefusal("emit", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  {.emit: "y[0] = 1.0;".}"""), "gpuEmit")

expectRefusal("inline_asm", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  asm "nop\""""), "gpuInlineAsm")

expectRefusal("midkernel_return", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  if a[0] > 0.0'f32:
    return
  y[0] = 1.0'f32"""), "gpuReturn")

expectRefusal("addr_of", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  if addr a[0] == addr a[1]:
    discard
  y[0] = 1.0'f32"""), "gpuAddr")

expectRefusal("ptr_local", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  let p = addr a[0]
  y[0] = p[]"""), "ptr Float32")

expectRefusal("bitcast", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  let t = cast[int32](a[0])
  y[0] = t.float32"""), "gpuCast")

# ── Member and aggregate rows (objects come from the snippet prelude) ─────

const objPrelude = """
type V = object
  x: float32
"""

expectRefusal("member_access", objPrelude & entryBody(
  "o: V, y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  y[0] = o.x"""), "gpuDot")

expectRefusal("array_literal", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  y[0] = [1.0'f32, 2.0'f32][0]"""), "gpuArrayLit")

# ── gpuCall policy: pure callees, barriers, atomics ───────────────────────

expectRefusal("pure_callee", """
import workspace/crucible

proc scale(x: float32): float32 =
  x * 2.0'f32

proc kernel(y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32) {.global.} =
  for i in 0 ..< n:
    y[i] = scale(a[i])

const pair = bkGlsl.bwd_diff(kernel)
""", "gpuCall")

expectRefusal("barrier", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  threadgroup_barrier()
  y[0] = a[0]"""), "gpuCall")

expectRefusal("atomic", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  y[0] = atomicFetchAdd(addr y[0], 1.0'f32)"""), "atomicFetchAdd")

# ── Differentiated-value contracts ────────────────────────────────────────

expectRefusal("comparison_value", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  y[0] = (a[0] > a[1]).float32 * 2.0'f32"""), "gpuConv")

expectRefusal("int_cast_value", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  y[0] = a[0].int32.float32"""), "gpuConv")

expectRefusal("compound_assign", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  y[0] += 1.0'f32"""), "gpuAssign")

expectRefusal("loop_carried", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  var acc = 0.0'f32
  for i in 0 ..< n:
    acc = acc + a[i]
  y[0] = acc"""), "loop-carried")

expectRefusal("inplace_read", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  for i in 0 ..< n:
    y[i] = y[i] + a[i]"""), "out-of-place")

expectRefusal("nested_loops", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  for i in 0 ..< n:
    for j in 0 ..< n:
      y[i] = a[i] + a[j]"""), "nested loops")

expectRefusal("inclusive_range", entryBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """  for i in 0 .. n - 1:
    y[i] = a[i]"""), "exclusive ranges")

expectRefusal("scalar_outside_loop", entryBody(
  "y: ptr UncheckedArray[float32], alpha: float32, a: ptr UncheckedArray[float32]",
  """  y[0] = alpha * a[0]"""), "outside the index loop")

# ── Entry contracts ───────────────────────────────────────────────────────

expectRefusal("not_a_proc", """
import workspace/crucible

const value = 42
const pair = bkMetal.bwd_diff(value)
""", "is not a proc")

expectRefusal("unsupported_backend", """
import workspace/crucible

proc kernel(y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32) {.global.} =
  for i in 0 ..< n:
    y[i] = a[i]

const pair = bkCuda.bwd_diff(kernel)
""", "no autodiff pipeline")
