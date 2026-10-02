## Run:
##   nim ceramic_cgs runner=cgs_layout_concat dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# direct concat of two tuple layouts into one layout
const concatDirectMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc concatDirectKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = make_layout(concat((2, 3), (4,)), concat((1, 3), (12,)))
    C[0] = float32 toIntVal crd2idx(r, (1, 1, 1))

# ── kernel rows ──

cgsReport("cgs_layout_concat", [
  cgsReceipt("concatDirectKernel", concatDirectMsl)])
