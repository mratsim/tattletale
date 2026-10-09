## explodePointerStructParams rewrites field chains rooted at an indexed or called expression:
## an exploded param referenced inside the index expression still rewrites.
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/ir/legalizations/test_explode_indexed_root.nim (from tattletale)

import std/[algorithm, macros, sequtils, sets, strutils, tables]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/codegen/ir/nim_to_gpu
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/passes/passes_legalizations

type
  Layout2D = object
    shape: tuple[rows, cols: int32]
    stride: tuple[rowS, colS: int32]

  View[T] = object
    data: ptr UncheckedArray[T]
    layout: Layout2D

  Vec2 = object
    x, y: float32

proc collectIdents(n: GpuAst, ids: var HashSet[string]) =
  if n == nil: return
  if n.kind == gpuIdent and n.symbol != nil:
    ids.incl n.symbol.iSym
  for child in n.items:
    collectIdents(child, ids)

macro bodyIdents(body: typed): string =
  ## Runs the body through the legalization passes and reports the kernel
  ## body's identifier set.
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
      var ids: HashSet[string]
      collectIdents(fn.pBody, ids)
      report.add fn.pName.ident() & ": " & toSeq(ids).sorted().join(", ")
  result = newLit(report)

proc runTest() =
  const idents = bodyIdents:
    proc idxKernel(outArr: ptr UncheckedArray[float32], dst: View[float32], boxes: ptr UncheckedArray[Vec2]) {.global.} =
      let y = boxes[dst.layout.shape.rows].y
      outArr[0] = y + dst.data[0]

  # `dst` is referenced inside the index expression of `boxes[...]`, whose
  # chain root is a gpuIndex, not an exploded param: the descent must still
  # rewrite `dst.layout.shape.rows` onto the exploded leaf
  doAssert "dst_layout" in idents, idents
  doAssert "dst_data" in idents, idents
  doAssert not idents.split(", ").anyIt(it == "dst" or it.startsWith("dst___")),
    "bare struct param 'dst' survived an indexed-root chain:\n" & idents

when isMainModule:
  runTest()
