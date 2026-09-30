## Codesize ledger, concat family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## cgsReport renders cost of 1 call plus the marginal over the paired floor.
## No floor pairs exist in this family, the Marginal column prints `-` throughout.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the concat track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# direct concat of two tuple layouts into one layout
const concatDirectMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc concatDirectKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = make_layout(concat((2, 3), (4,)), concat((1, 3), (12,)))
    C[0] = float32 toIntVal crd2idx(r, (1, 1, 1))

# ── kernel rows ──

cgsReport("cgs_layout_concat", [
  cgsReceipt("concatDirectKernel", concatDirectMsl)])
