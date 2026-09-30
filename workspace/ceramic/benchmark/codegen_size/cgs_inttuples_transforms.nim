## Codesize ledger, int-tuple transform family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## concat and flatten get a runtime row plus a static row, unwrap one runtime row.
##
## select carries no rows
## - its macro emits 64-bit-literal bracket indices
## - metal converter rejects them, unmeasurable without a src change
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# concat over two runtime tuples, tuple + tuple macro overload, block-dimension append call shape
const concatTupleTupleMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc concatTupleTupleKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let r = concat((int(M), int(N)), (16, 16))
    C[0] = float32 toIntVal(r[0])

# concat over a static leading int and a runtime tuple, prepend-a-static-dimension call shape
const concatStaticTupleMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc concatStaticTupleKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let r = concat(16, (int(M), int(N)))
    C[0] = float32 toIntVal(r[1])

# flatten over a runtime nested tuple, leaf-passthrough body, coalesce flatten(shape) call shape
const flattenRuntimeMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc flattenRuntimeKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let t = ((int(M), 16), (int(N), 16))
    let r = flatten(t)
    C[0] = float32 toIntVal(r[1])

# flatten over a fully static tuple, static overload still emits the inline proc over a runtime-typed let
const flattenStaticMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc flattenStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = flatten((16, (32, 64)))
    C[0] = float32 toIntVal(r[1])

# unwrap over a runtime rank-1 tuple, 1-element collapse call shape
const unwrapMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc unwrapKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let r = unwrap((int(M),))
    C[0] = float32 r

# ── kernel rows ──

cgsReport("cgs_inttuples_transforms", [
  cgsReceipt("concatTupleTupleKernel", concatTupleTupleMsl),
  cgsReceipt("concatStaticTupleKernel", concatStaticTupleMsl),
  cgsReceipt("flattenRuntimeKernel", flattenRuntimeMsl),
  cgsReceipt("flattenStaticKernel", flattenStaticMsl),
  cgsReceipt("unwrapKernel", unwrapMsl)])
