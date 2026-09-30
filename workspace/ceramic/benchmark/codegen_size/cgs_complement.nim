## Codesize ledger, complement family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible
## metal backend and prints its MSL byte size.
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## Usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the complement track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_layout_copy_gpu

# complement of a runtime-shape rank-2 layout, static strides, the direct-call pattern of tests/test_layout_algebra.nim
const complementDirectMsl = metal:
  proc complementDirectKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    let r = complement(L)
    C[0] = float32 toIntVal size(r)

# complement of a fully static layout, the compile-time path
const complementStaticMsl = metal:
  proc complementStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 2), (1, 16))
    let r = complement(L)
    C[0] = float32 toIntVal size(r)

# logical_divide with a Layout tiler, the complement + compose general path
const logicalDivideCompMsl = metal:
  proc logicalDivideCompKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    let d = logical_divide(L, make_layout((16, 4)))
    C[0] = float32 toIntVal size(d)

# logical_product, the compose(complement(a, ...), tiler) site
const logicalProductCompMsl = metal:
  proc logicalProductCompKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    let p = logical_product(L, make_layout((16, 16)))
    C[0] = float32 toIntVal size(p)

# the copy chain, right_inverse + coalesce(compose(dst, R)) in copyFrom,
# the KV-write pattern, row-padded dst and row-compact src
const copyChainMsl = metal:
  proc copyChainKernel(D, S: ptr UncheckedArray[float32]; M, N, Sd, Ss: int32) {.global.} =
    var dst = make_view(D, (int(M), int(N)), (1, int(Sd)))
    let src = make_view(S, (int(M), int(N)), (1, int(Ss)))
    copyFrom(dst, src)

# local_tile, complement via zipped_divide -> logical_divide
const localTileCompMsl = metal:
  proc localTileCompKernel(C: ptr UncheckedArray[float32]; M, N, i, j: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(N)))
    let t = local_tile(p, (16, 16), (int(i), int(j)))
    C[0] = t(0, 0)

echo "complementDirectKernel: ", complementDirectMsl.len
echo "complementStaticKernel: ", complementStaticMsl.len
echo "logicalDivideCompKernel: ", logicalDivideCompMsl.len
echo "logicalProductCompKernel: ", logicalProductCompMsl.len
echo "copyChainKernel: ", copyChainMsl.len
echo "localTileCompKernel: ", localTileCompMsl.len
