## Run:
##   nim ceramic_cgs runner=cgs_tensor_selectors dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

{.experimental: "callOperator".}

# ── baseline, no selector machinery ──

# dynamic rank-2 view, one manual-offset element read
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p.data[int R0 * 1 + int C0 * int S]

# ── `()` call operator ──

# element read branch, the lvalue path (tensor_selectors.nim:32)
const callOperatorReadMsl = metal:
  proc callOperatorReadKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p(int R0, int C0)

# element assignment branch, the lvalue written through `()` (tensor_selectors.nim:38)
const callOperatorAssignMsl = metal:
  proc callOperatorAssignKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    p(int R0, int C0) = 1.5'f32

# underscore branch, the sub-View path (tensor_selectors.nim:25)
const callOperatorSubviewMsl = metal:
  proc callOperatorSubviewKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let s = p(int R0, _)
    C[0] = float32 s.data[0]

# ── `[]` element access ──

const elementReadMsl = metal:
  proc elementReadKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p[int R0, int C0]

const elementWriteMsl = metal:
  proc elementWriteKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    p[int R0, int C0] = 1.5'f32

# ── slice, subtensor via underscore dispatch ──

const sliceMsl = metal:
  proc sliceKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let s = slice(p, (int R0, _))
    C[0] = float32 s(0, int C0)

# ── displace ──

const displaceViewMsl = metal:
  proc displaceViewKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    let d = p.displace((int R0, int C0))
    C[0] = float32 d(0, 0)

# TensorOwned row: the owned tensor carries its own stack array,
# the displacement runs through t.view() per the API split
const displaceOwnedMsl = metal:
  proc displaceOwnedKernel(C: ptr UncheckedArray[float32]; R0, C0: int32) {.global.} =
    var t = make_tensor(float32, (16, 16), (1, 16))
    let d = t.displace((int R0, int C0))
    C[0] = float32 d(0, 0)

# ── kernel rows ──
# Marginal = cost minus baseline, computed inside cgsReport, every row
# sits on baselineOverheadKernel except the owned-tensor row, which has
# no view baseline
cgsReport("cgs_tensor_selectors", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("callOperatorReadKernel", callOperatorReadMsl, baselineOverheadMsl.len),
  cgsReceipt("callOperatorAssignKernel", callOperatorAssignMsl, baselineOverheadMsl.len),
  cgsReceipt("callOperatorSubviewKernel", callOperatorSubviewMsl, baselineOverheadMsl.len),
  cgsReceipt("elementReadKernel", elementReadMsl, baselineOverheadMsl.len),
  cgsReceipt("elementWriteKernel", elementWriteMsl, baselineOverheadMsl.len),
  cgsReceipt("sliceKernel", sliceMsl, baselineOverheadMsl.len),
  cgsReceipt("displaceViewKernel", displaceViewMsl, baselineOverheadMsl.len),
  cgsReceipt("displaceOwnedKernel", displaceOwnedMsl)])
