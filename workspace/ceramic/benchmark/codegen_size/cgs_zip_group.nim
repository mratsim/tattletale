## Codesize ledger, zip/group family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible
## metal backend and prints its MSL byte size.
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## Usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the zip/group track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

# zipped_divide with a tuple tiler on a runtime rank-2 layout
const zippedDivideTupleMsl = metal:
  proc zippedDivideTupleKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, int(N)))
    let zd = zipped_divide(L, (16, 16))
    C[0] = float32 toIntVal size(zd)

# zipped_divide with a Layout tiler, the tile_unzip general path
const zippedDivideLayoutMsl = metal:
  proc zippedDivideLayoutKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, int(N)))
    let zd = zipped_divide(L, make_layout((16, 4), (1, 4)))
    C[0] = float32 toIntVal size(zd)

# zipped_divide on a runtime rank-4 layout, the NHWC gmem view pattern
const zippedDivideRank4Msl = metal:
  proc zippedDivideRank4Kernel(C: ptr UncheckedArray[float32]; N4, C4, H4, W4: int32) {.global.} =
    let L = make_layout((int(N4), int(C4), int(H4), int(W4)),
                        (int(C4 * H4 * W4), int(H4 * W4), int(W4), 1))
    let zd = zipped_divide(L, (8, 8, 8, 8))
    C[0] = float32 toIntVal size(zd)

# groupDimensions wrapping dimensions [0, 2) of a static rank-4 layout
const groupDimensionsMsl = metal:
  proc groupDimensionsKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 3, 5, 7))
    let g = groupDimensions(L, 0, 2)
    C[0] = float32 toIntVal crd2idx(g, ((1, 2), 1, 1))

# zipDimensions interleaving two static rank-2 layouts
const zipDimensionsMsl = metal:
  proc zipDimensionsKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((2, 3), (1, 3))
    let b = make_layout((2, 3), (6, 2))
    let z = zipDimensions(a, b)
    C[0] = float32 toIntVal crd2idx(z, ((1, 1), (1, 2)))

echo "zippedDivideTupleKernel: ", zippedDivideTupleMsl.len
echo "zippedDivideLayoutKernel: ", zippedDivideLayoutMsl.len
echo "zippedDivideRank4Kernel: ", zippedDivideRank4Msl.len
echo "groupDimensionsKernel: ", groupDimensionsMsl.len
echo "zipDimensionsKernel: ", zipDimensionsMsl.len
