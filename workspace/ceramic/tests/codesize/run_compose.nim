## Codesize ledger, compose family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible
## metal backend and prints its MSL byte size.
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## the command is in tests/codesize/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the compose track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# compose of two static rank-2 layouts, the composeImpl path
const composeStaticMsl = metal:
  proc composeStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((4, 4), (1, 4))
    let b = make_layout((2, 2), (1, 2))
    let r = compose(a, b)
    C[0] = float32 toIntVal crd2idx(r, (1, 1))

# compose of a static rank-2 layout with a nested layout, the composeDistribute path
const composeNestedMsl = metal:
  proc composeNestedKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((8, 8), (1, 8))
    let b = make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8)))
    let r = compose(a, b)
    C[0] = float32 toIntVal crd2idx(r, ((1, 1), (1, 1)))

# compose with a runtime rank-1 LHS, the b-strides scaleBy path
const composeRank1Msl = metal:
  proc composeRank1Kernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let a = make_layout(int(M), 2)
    let b = make_layout((4, 4), (1, 4))
    let r = compose(a, b)
    C[0] = float32 toIntVal crd2idx(r, (1, 1))

# compose of a runtime layout with a static layout, the thrfrg_A/B/C call-site pattern
const composeDynMsl = metal:
  proc composeDynKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let a = make_layout((int(M), int(N)), (1, 16))
    let b = make_layout((4, 4), (1, 4))
    let r = compose(a, b)
    C[0] = float32 toIntVal crd2idx(r, (1, 1))

echo "composeStaticKernel: ", cstring(composeStaticMsl).len
echo "composeNestedKernel: ", cstring(composeNestedMsl).len
echo "composeRank1Kernel: ", cstring(composeRank1Msl).len
echo "composeDynKernel: ", cstring(composeDynMsl).len
