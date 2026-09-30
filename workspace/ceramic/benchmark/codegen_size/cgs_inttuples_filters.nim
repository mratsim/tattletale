## Codesize ledger, int-tuple filter family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## slice gets one row over the slice call shape, X/Y selectors against a runtime 3-tuple.
## tupleType, tupleTypeLen, and the stream/builder plumbing are compile-time only, zero-emission, no rows.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# slice over a runtime 3-tuple against X/Y selectors
# call-site body, type predicates decide at compile time, kept leaves are
# read from runtime values
const filterZipMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc filterZipKernel(C: ptr UncheckedArray[float32]; M, N, K: int32) {.global.} =
    let t = (int(M), int(N), int(K))
    let r = slice(t, (X(), Y(), X()))
    C[0] = float32 toIntVal(r[1])

# ── kernel rows ──

cgsReport("cgs_inttuples_filters", [
  cgsReceipt("filterZipKernel", filterZipMsl)])
