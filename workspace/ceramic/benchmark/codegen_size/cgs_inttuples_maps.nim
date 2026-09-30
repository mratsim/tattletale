## Codesize ledger, int-tuple map family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## mapLeavesWith, mapDimensionsWith, flatMapLeaves, and concatFlat get one runtime row each over runtime leaf values.
##
## Compile-time-only ops of the module carry no rows
## - flatLeaves takes NimNode
## - countLeaves folds to a literal
## - mapLeavesWith identity body returns t verbatim
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# mapLeavesWith over a runtime leaf map, scaleBy call-site shape, divide-chain pair-tuple input, leaf read keeps the row
const mapLeavesMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc mapLeavesKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let t = ((int(M div 16), 16), (int(N div 16), 16))
    let r = mapLeavesWith(t):
      it * int(S)
    C[0] = float32 toIntVal(r[0][0])

# mapDimensionsWith over a top-level product map, product_each call-site shape, runtime nested tuples
const mapDimensionsMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc mapDimensionsKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let t = ((int(M), 16), (int(N), 16))
    let r = mapDimensionsWith(t):
      product(it)
    C[0] = float32 toIntVal(r[0])

# flatMapLeaves into the flat pack with a real leaf body, generic leaf map flatten cannot express
const flatMapLeavesMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc flatMapLeavesKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let t = ((int(M), 16), (int(N), 16))
    let r = flatMapLeaves(t):
      it * 2
    C[0] = float32 toIntVal(r[2])

# concatFlat over two runtime dimension pairs, leaf stream of a then of b into one flat tuple
const concatFlatMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc concatFlatKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let r = concatFlat((int(M), 16), (int(N), 16))
    C[0] = float32 toIntVal(r[1])

# ── kernel rows ──

cgsReport("cgs_inttuples_maps", [
  cgsReceipt("mapLeavesKernel", mapLeavesMsl),
  cgsReceipt("mapDimensionsKernel", mapDimensionsMsl),
  cgsReceipt("flatMapLeavesKernel", flatMapLeavesMsl),
  cgsReceipt("concatFlatKernel", concatFlatMsl)])
