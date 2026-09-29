## Codesize ledger, coalesce family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible
## metal backend and prints its MSL byte size.
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## Usage in experiments/codesize/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the coalesce track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# coalesce of a static contiguous rank-3 layout
const coalesceStaticMsl = metal:
  proc coalesceStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 4, 8), (1, 2, 8))
    let r = coalesce(L)
    C[0] = float32 toIntVal crd2idx(r, 7)

# coalesce of a static layout with a stride-0 trailing dimension
const coalesceZerosMsl = metal:
  proc coalesceZerosKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 1), (1, 0))
    let r = coalesce(L)
    C[0] = float32 toIntVal crd2idx(r, 2)

# filter_inactive, the coalesce(filter_zeros(...)) chain
const filterInactiveMsl = metal:
  proc filterInactiveKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 1), (1, 0))
    let r = filter_inactive(L)
    C[0] = float32 toIntVal crd2idx(r, 2)

# coalesce of a runtime layout, the copy-chain pattern
const coalesceDynMsl = metal:
  proc coalesceDynKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(S)))
    let r = coalesce(p.layout)
    C[0] = float32 toIntVal size(r)

echo "coalesceStaticKernel: ", cstring(coalesceStaticMsl).len
echo "coalesceZerosKernel: ", cstring(coalesceZerosMsl).len
echo "filterInactiveKernel: ", cstring(filterInactiveMsl).len
echo "coalesceDynKernel: ", cstring(coalesceDynMsl).len
