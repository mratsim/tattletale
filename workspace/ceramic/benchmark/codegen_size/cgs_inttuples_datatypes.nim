## Codesize ledger, int-tuple datatype family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## ceil_div, makeIntTuple, and sign get one runtime row each, runtime-emitting forms only.
##
## Compile-time-only forms carry no rows
## - toIntVal folds, identity over int and constant over Int[V]
## - rank and the Int[N]-typed operator overloads are static, they fold to constants
## - makeIntTuple is defined in int_tuples.nim, measured here per the dependency map
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# ceil_div over runtime ints, gap computation for the complement and pad paths
const ceilDivMsl = metal:
  proc ceilDivKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    C[0] = float32 ceil_div(int(M), 16)

# makeIntTuple over a mixed static/runtime tuple, coord-wrap site crd2idx goes through, static leaf wraps to Int[N], runtime leaf stays int
const makeIntTupleMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc makeIntTupleKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let r = makeIntTuple((16, int(M)))
    C[0] = float32 (toIntVal r[0]) + r[1]

# sign over a runtime stride value, broadcast-direction call-site shape
const signMsl = metal:
  proc signKernel(C: ptr UncheckedArray[float32]; S: int32) {.global.} =
    C[0] = float32 sign(int(S))

# ── kernel rows ──

cgsReport("cgs_inttuples_datatypes", [
  cgsReceipt("ceilDivKernel", ceilDivMsl),
  cgsReceipt("makeIntTupleKernel", makeIntTupleMsl),
  cgsReceipt("signKernel", signMsl)])
