## IR contract of the `explodePointerStructParams` pass: struct kernel params carrying
## pointer fields lower to one param per leaf field, body field accesses rewrite to them.
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/test_explode_struct_params.nim (from tattletale)

import std/[algorithm, macros, sequtils, sets, strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/nim_to_gpu
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/passes/passes_legalizations

# ── Recreated Ceramic types (crucible-only) ─────────────────────────────────

type
  StoreMask {.size: sizeof(uint32).} = enum
    smStoreNone, smStoreAll, smStoreDiag

  Layout2D = object
    shape: tuple[rows, cols: int32]
    stride: tuple[rowS, colS: int32]

  View[T] = object
    ## TensorView stand-in: device pointer + layout value
    data: ptr UncheckedArray[T]
    layout: Layout2D

  EpiAXPBY[T] = object
    ## D = alpha*x + beta*c, gated by storeMask
    alpha, beta: T
    c: View[T]
    storeMask: StoreMask

  Vec2 = object
    x, y: uint32

# ── Introspection helpers (compile-time) ────────────────────────────────────

proc typeStr(t: GpuType): string =
  case t.kind
  of gtPtr:
    let inner = if t.to.kind == gtUA: t.to.uaTo else: t.to
    result = "ptr<" & typeStr(inner) & ">"
  of gtUA: result = "ua<" & typeStr(t.uaTo) & ">"
  of gtGenericInst:
    result = t.gName
    for g in t.gArgs: result.add "<" & typeStr(g) & ">"
  of gtObject: result = t.name
  of gtBool: result = "bool"
  of gtInt32: result = "int32"
  of gtUint32: result = "uint32"
  of gtFloat32: result = "float32"
  else: result = $t.kind

proc collectIdents(n: GpuAst, ids: var HashSet[string]) =
  if n == nil: return
  if n.kind == gpuIdent and n.symbol != nil:
    ids.incl n.symbol.iSym
  for child in n.items:
    collectIdents(child, ids)

macro explodeReport(body: typed): string =
  ## Runs the body through the legalization passes only and returns a report:
  ## per kernel, the exploded parameter list and the body's identifier set.
  var ctx = GpuContext()
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerExplodePointerStructParams()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toGpuAst(typeReg, body)
  ctx.types = typeReg.types
  runPasses(ctx, reg)
  var report = ""
  for fnKey, fn in ctx.allFnTab:
    if fn.kind == gpuProc and attGlobal in fn.pAttributes:
      report.add "kernel " & fn.pName.ident() & "\n"
      for i, p in fn.pParams:
        report.add "  param " & $i & ": " & p.ident.ident() &
          " : " & typeStr(p.typ) & "\n"
      var ids: HashSet[string]
      collectIdents(fn.pBody, ids)
      let sorted = toSeq(ids).sorted()
      report.add "  body idents: " & sorted.join(", ") & "\n"
  result = newLit(report)

const irReport = explodeReport:
  proc axpbyKernel(dst: View[float32], x: View[float32], epi: EpiAXPBY[float32]) {.global.} =
    let rows = dst.layout.shape.rows
    let cols = dst.layout.shape.cols
    for r in 0 ..< rows:
      for c in 0 ..< cols:
        let i = r * dst.layout.stride.rowS + c * dst.layout.stride.colS
        let xi = r * x.layout.stride.rowS + c * x.layout.stride.colS
        let ci = r * epi.c.layout.stride.rowS + c * epi.c.layout.stride.colS
        if uint32(epi.storeMask) == 1'u32:
          dst.data[i] = epi.alpha * x.data[xi] + epi.beta * epi.c.data[ci]
        else:
          dst.data[i] = 0.0'f32

  proc sumKernel(v: Vec2, outArr: ptr UncheckedArray[uint32]) {.global.} =
    outArr[0] = v.x + v.y

proc axpbyIdents(report: string): seq[string] =
  ## axpbyKernel body identifier set from the report.
  var inAxpby = false
  for line in report.splitLines():
    if line.startsWith("kernel "):
      inAxpby = line == "kernel axpbyKernel"
    elif inAxpby and line.startsWith("  body idents:"):
      result = line["  body idents: ".len .. ^1].split(", ")

# ── Contract checks ──────────────────────────────────────────────────────────

proc runTest() =
  # The full report is the failure evidence, so print it once up front.
  echo irReport

  let expectedParams = [
    "param 0: dst_data : ptr<float32>",
    "param 1: dst_layout : Layout2D",
    "param 2: x_data : ptr<float32>",
    "param 3: x_layout : Layout2D",
    "param 4: epi_alpha : float32",
    "param 5: epi_beta : float32",
    "param 6: epi_c_data : ptr<float32>",
    "param 7: epi_c_layout : Layout2D",
    "param 8: epi_storeMask : uint32",
  ]
  for line in expectedParams:
    doAssert line in irReport, "missing exploded param '" & line & "' in:\n" & irReport

  doAssert ": View<float32>" notin irReport,
    "whole View param survived:\n" & irReport
  doAssert ": EpiAXPBY<float32>" notin irReport,
    "whole EpiAXPBY param survived:\n" & irReport

  let idents = axpbyIdents(irReport)
  let expectedLeaves = ["dst_data", "dst_layout", "x_data", "x_layout",
                        "epi_alpha", "epi_beta", "epi_c_data", "epi_c_layout",
                        "epi_storeMask"]
  for leaf in expectedLeaves:
    doAssert leaf in idents,
      "exploded leaf '" & leaf & "' not used in the body, idents: " & $idents
  # the original struct param names must not survive, neither in bare
  # form nor gensym-suffixed (iSym = name & "___" & hash)
  for bare in ["dst", "x", "epi"]:
    for id in idents:
      doAssert id != bare and not id.startsWith(bare & "___"),
        "bare struct param '" & bare & "' survived as '" & id & "'"

  # pointer-free struct param stays whole
  doAssert "param 0: v : Vec2" in irReport,
    "pointer-free Vec2 param was not left whole:\n" & irReport

  # CUDA and OpenCL keep struct params whole (no explosion wired)
  block:
    const cudaSrc = cuda:
      proc axpbyKernelCuda(dst: View[float32], x: View[float32], epi: EpiAXPBY[float32]) {.global.} =
        let n = dst.layout.shape.rows * dst.layout.shape.cols
        for i in 0 ..< n:
          if uint32(epi.storeMask) == 1'u32:
            dst.data[i] = epi.alpha * x.data[i] + epi.beta * epi.c.data[i]
          else:
            dst.data[i] = 0.0'f32
    doAssert "EpiAXPBYf32& epi" in cudaSrc,
      "CUDA should take the struct param natively, got:\n" & cudaSrc
    doAssert "epi_alpha" notin cudaSrc,
      "CUDA source must not contain exploded params, got:\n" & cudaSrc

    const oclSrc = opencl:
      proc axpbyKernelOcl(dst: View[float32], x: View[float32], epi: EpiAXPBY[float32]) {.global.} =
        let n = dst.layout.shape.rows * dst.layout.shape.cols
        for i in 0 ..< n:
          if uint32(epi.storeMask) == 1'u32:
            dst.data[i] = epi.alpha * x.data[i] + epi.beta * epi.c.data[i]
          else:
            dst.data[i] = 0.0'f32
    doAssert "EpiAXPBYf32 epi" in oclSrc,
      "OpenCL should take the struct param natively, got:\n" & oclSrc
    doAssert "epi_alpha" notin oclSrc,
      "OpenCL source must not contain exploded params, got:\n" & oclSrc

when isMainModule:
  runTest()
