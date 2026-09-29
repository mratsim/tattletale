## Codesize ledger, like-constructors family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible
## metal backend and prints its MSL byte size.
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## Usage in experiments/codesize/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the like-constructors track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# make_layout_like on a compacting static rank-2 layout (2,3):(2,1) -> (3,1)
const layoutLikeCompactMsl = metal:
  proc layoutLikeCompactKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 3), (2, 1))
    let r = make_layout_like(L)
    C[0] = float32 toIntVal size(r)

# make_layout_like on a runtime-stride layout (the make_tensor_like site)
const layoutLikeDynStrideMsl = metal:
  proc layoutLikeDynStrideKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(S)))
    let r = make_layout_like(p.layout)
    C[0] = float32 toIntVal size(r)

# make_fragment_like with a (16, 2) V block, the tensor-core fragment call site
const fragmentLikeVMsl = metal:
  proc fragmentLikeVKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout(((16, 2), 16), ((1, 16), 4))
    let f = make_fragment_like(L)
    C[0] = float32 toIntVal size(f)

# make_fragment_like with a broadcast V, the epilogue broadcast-bias call site
const fragmentLikeBroadcastMsl = metal:
  proc fragmentLikeBroadcastKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 8), (0, 1))
    let f = make_fragment_like(L)
    C[0] = float32 toIntVal size(f)

echo "layoutLikeCompactKernel: ", cstring(layoutLikeCompactMsl).len
echo "layoutLikeDynStrideKernel: ", cstring(layoutLikeDynStrideMsl).len
echo "fragmentLikeVKernel: ", cstring(fragmentLikeVMsl).len
echo "fragmentLikeBroadcastKernel: ", cstring(fragmentLikeBroadcastMsl).len
