# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## bwd_diff refusal suite
##
## Test that every construct outside the differentiable subset fails
## compilation with a `bwd_diff:`-prefixed diagnostic naming the construct:
##
## - one auto-runnable case per refused row of the node-kind closure table
## - plus the structural contracts: compound assignment, loop-carried overwrites, out-of-place reads, nested loops, scalar cotangents
## - each case compiles a standalone snippet through `nim check` and must fail
##   with the prefix and the offending construct in the message
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/autodiff/t_ir_bwd_diff_refusals.nim

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
  let dir = repoRoot() / "build" / "bwd_refusals"
  createDir(dir)
  let path = dir / ("case_" & name & ".nim")
  writeFile(path, src)
  let cmd = "nim check --hints:off --warnings:off --verbosity:0 --nimcache:nimcache/bwdrefusals --path:. " & path
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

proc kernelBody(params: string, body: string): string =
  ## A standalone snippet that runs the full AD pipeline over one kernel.
  result = """
import workspace/crucible/tests/autodiff/ad_ir_testutils

static:
  discard adKernels:
    proc kernel(""" & params & """) {.global.} =
""" & body

# ── Refused node kinds (one case per refused row) ─────────────────────────

expectRefusal("while", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      var j = 0.0'f32
      while j < 4.0'f32:
        j = j + 1.0'f32"""), "gpuWhile")

expectRefusal("emit", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      {.emit: "y[0] = 1.0;".}"""), "gpuEmit")

expectRefusal("inline_asm", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      asm "nop\""""), "gpuInlineAsm")

expectRefusal("midkernel_return", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      if a[0] > 0.0'f32:
        return
      y[0] = 1.0'f32"""), "gpuReturn")

expectRefusal("addr_of", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      if addr a[0] == addr a[1]:
        discard
      y[0] = 1.0'f32"""), "gpuAddr")

expectRefusal("ptr_local", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      let p = addr a[0]
      y[0] = p[]"""), "ptr Float32")

expectRefusal("bitcast", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      let t = cast[int32](a[0])
      y[0] = t.float32"""), "gpuCast")

# ── Member and aggregate rows (objects come from the snippet prelude) ─────

const objPrelude = """
type V = object
  x: float32
"""

expectRefusal("member_access", objPrelude & kernelBody(
  "o: V, y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      y[0] = o.x"""), "gpuDot")

# gpuObjConstr is unreachable through the frontend: the frontend eliminates
# scalar-field constructions before the AD pass runs.
# The row is driven through the registered pass on a hand-built fixture kernel.
expectRefusal("object_constr", """
import std/tables
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_autodiff

static:
  var ctx = GpuContext()
  let f32 = GpuType(kind: gtFloat32)
  let bufTyp = GpuType(kind: gtPtr,
                       to: GpuType(kind: gtUncheckedArray, uaTo: f32),
                       addressSpace: asDevice)
  let objTyp = GpuType(kind: gtObject, name: "V")
  objTyp.fields = @[GpuTypeField(name: "x", typ: f32)]
  let yIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(
    "y", iSym = "y_h", typ = bufTyp, symKind = gsDeviceKernelParam))
  let constr = GpuAst(kind: gpuObjConstr, ocType: objTyp,
    ocFields: @[GpuFieldInit(name: "x",
                             value: GpuAst(kind: gpuLit, lValue: "1", lType: f32),
                             typ: f32)])
  let assign = GpuAst(kind: gpuAssign,
    aLeft: GpuAst(kind: gpuIndex, iArr: yIdent,
                  iIndex: GpuAst(kind: gpuLit, lValue: "0",
                                 lType: GpuType(kind: gtInt32))),
    aRight: constr, aOp: "=")
  var fn = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent, symbol: newSymbol("k", iSym = "k_h", symKind = gsProc)),
    pRetType: GpuType(kind: gtVoid),
    pParams: @[GpuParam(ident: yIdent, typ: bufTyp, addressSpace: asDevice,
                        passByRef: false, flavor: kIn)],
    pBody: GpuAst(kind: gpuBlock, statements: @[assign]),
    pAttributes: {attGlobal})
  ctx.allFnTab[fn.pName] = fn
  var reg = PassRegistry.new()
  reg.registerAutodiffPasses()
  ctx.runPasses(reg)
""", "gpuObjConstr")

expectRefusal("array_literal", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      y[0] = [1.0'f32, 2.0'f32][0]"""), "gpuArrayLit")

# ── gpuCall policy: pure callees, barriers, atomics ───────────────────────

expectRefusal("pure_callee", """
import workspace/crucible/tests/autodiff/ad_ir_testutils

static:
  discard adKernels:
    proc scale(x: float32): float32 =
      x * 2.0'f32
    proc kernel(y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32) {.global.} =
      for i in 0 ..< n:
        y[i] = scale(a[i])
""", "gpuCall")

expectRefusal("barrier", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      threadgroup_barrier()
      y[0] = a[0]"""), "gpuCall")

expectRefusal("atomic", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      y[0] = atomicFetchAdd(addr y[0], 1.0'f32)"""), "atomicFetchAdd")

# ── Differentiated-value contracts ────────────────────────────────────────

expectRefusal("comparison_value", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      y[0] = (a[0] > a[1]).float32 * 2.0'f32"""), "gpuConv")

expectRefusal("int_cast_value", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      y[0] = a[0].int32.float32"""), "gpuConv")

expectRefusal("compound_assign", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      y[0] += 1.0'f32"""), "gpuAssign")

expectRefusal("loop_carried", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      var acc = 0.0'f32
      for i in 0 ..< n:
        acc = acc + a[i]
      y[0] = acc"""), "loop-carried")

expectRefusal("inplace_read", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      for i in 0 ..< n:
        y[i] = y[i] + a[i]"""), "out-of-place")

expectRefusal("nested_loops", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      for i in 0 ..< n:
        for j in 0 ..< n:
          y[i] = a[i] + a[j]"""), "nested loops")

expectRefusal("inclusive_range", kernelBody(
  "y: ptr UncheckedArray[float32], a: ptr UncheckedArray[float32], n: int32",
  """      for i in 0 .. n - 1:
        y[i] = a[i]"""), "exclusive ranges")

expectRefusal("scalar_outside_loop", kernelBody(
  "y: ptr UncheckedArray[float32], alpha: float32, a: ptr UncheckedArray[float32]",
  """      y[0] = alpha * a[0]"""), "outside the index loop")

echo "  OK — refusal battery: all cases refuse with the bwd_diff: prefix"
